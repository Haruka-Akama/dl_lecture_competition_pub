import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, Dataset, DistributedSampler
from torchmetrics import Accuracy
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
import torch.optim as optim
from termcolor import cprint
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torchvision import transforms
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import hydra
from hydra.core.hydra_config import HydraConfig
import json

from src.MEG_datasets import ThingsMEGDataset, ImageDataset
from src.MEG_models import fclip
from src.MEG_utils import set_seed

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    os.environ['NCCL_DEBUG'] = 'INFO'
    os.environ['NCCL_IB_DISABLE'] = '1'
    os.environ['NCCL_P2P_LEVEL'] = 'NVL'
    os.environ['NCCL_SHM_DISABLE'] = '1'
    os.environ['NCCL_LAUNCH_MODE'] = 'PARALLEL'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def run(rank, world_size, cfg_dict, hydra_config):
    setup(rank, world_size)
    cfg = OmegaConf.create(cfg_dict)
    
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')
    # 設定内容を表示


    # ログディレクトリの設定
    with open("hydra_config.json", "r") as f:
        hydra_cfg = json.load(f)
        logdir = hydra_cfg["hydra_run_dir"]
        
    if rank == 0:
        os.environ['HYDRA_RUN_DIR'] = logdir
    else:
        logdir = os.environ.get('HYDRA_RUN_DIR', logdir)
    logdir_list = [logdir]
    dist.broadcast_object_list(logdir_list, src=0)
    logdir = logdir_list[0]

    if cfg.use_wandb and rank == 0:
        wandb.init(mode="online", dir=logdir, project="MEG-classification")

    # ------------------
    #    Dataloader
    # ------------------
    loader_args = {"batch_size": cfg.batch_size // world_size, "num_workers": cfg.num_workers, "pin_memory": True, "persistent_workers": True}
    print(f"Rank {rank} Debug start")
    
    # データのサンプリング
    train_set = ThingsMEGDataset("train", cfg.data_dir)
    indices = torch.randperm(len(train_set))[:len(train_set) // 10]  # データセットの10%をランダムに選択
    train_set = Subset(train_set, indices)
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(train_set, shuffle=False, sampler=train_sampler, **loader_args)

    print(f"Rank {rank} Train load complete")
    
    val_set = ThingsMEGDataset("val", cfg.data_dir)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_set, num_replicas=world_size, rank=rank)
    val_loader = DataLoader(val_set, shuffle=False, sampler=val_sampler, **loader_args)

    print(f"Rank {rank} val load complete")

    test_set = ThingsMEGDataset("test", cfg.data_dir)
    test_loader = DataLoader(test_set, shuffle=False, batch_size=cfg.batch_size // world_size, num_workers=cfg.num_workers)
    
    print(f"Rank {rank} test load complete")

    # 画像データセット
    transform = transforms.Compose([transforms.ToTensor()])
    image_train_set = ImageDataset(split='train', images_dir='/data1/akamaharuka/Images', data_dir=cfg.data_dir, transform=transform)
    image_train_loader = DataLoader(image_train_set, shuffle=True, **loader_args)

    image_val_set = ImageDataset(split='val', images_dir='/data1/akamaharuka/Images', data_dir=cfg.data_dir, transform=transform)
    image_val_loader = DataLoader(image_val_set, shuffle=False, **loader_args)

    image_test_set = ImageDataset(split='test', images_dir='/data1/akamaharuka/Images', data_dir=cfg.data_dir, transform=transform)
    image_test_loader = DataLoader(image_test_set, shuffle=False, batch_size=cfg.batch_size, num_workers=cfg.num_workers)

    # ------------------
    #       Model
    # ------------------
    num_sensors = 306  # 仮のセンサー数
    sensor_locs = np.random.rand(num_sensors, 2)  # 2D位置
    sensor_locs = torch.tensor(sensor_locs, dtype=torch.float32)

    # 被験者IDの設定
    subject_ids = np.random.randint(0, 4, size=cfg.batch_size)  # 0から3までの被験者IDをランダムに生成
    subject_ids = torch.tensor(subject_ids, dtype=torch.long)

    # fclipモデルを初期化
    num_input_channels = 306  # 例として仮の入力チャネル数
    num_output_channels = 128  # 任意の出力チャネル数
    num_subjects = 4  # 仮の被験者数
    num_classes = 1854  # クラス数（画像カテゴリ数）
    model = fclip(num_input_channels, num_output_channels, num_subjects, num_classes).to(rank)

    # デバイス設定
   
    if torch.cuda.is_available():
        # 使用可能なGPUデバイスの名前を取得
        gpu_names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        print("Available GPU devices:")
        for idx, name in enumerate(gpu_names):
            print(f"  GPU {idx}: {name}")
        
        model = DDP(model, device_ids=[rank])
    else:
        print("Using CPU")
        model = model.to(device)

    # ------------------
    #     Optimizer
    # ------------------
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    step_scheduler = StepLR(optimizer, step_size=10, gamma=0.5)  # 10エポックごとに学習率を半減

    # ------------------
    #   Start training
    # ------------------  
    max_val_acc = 0
    accuracy = Accuracy(
        task="multiclass", num_classes=train_set.dataset.num_classes, top_k=10
    ).to(device)
    
    early_stopping_patience = 30
    early_stopping_counter = 0
    
    for epoch in range(cfg.epochs):
        print(f"Epoch {epoch+1}/{cfg.epochs} (Rank {rank})")
        
        # 明示的にメモリを解放
        torch.cuda.empty_cache()

        train_loss, train_acc, val_loss, val_acc = [], [], [], []
        
        model.train()
        for X, y, subject_idxs in train_loader:
            X, y, subject_idxs = X.to(device), y.to(device), subject_idxs.to(device)
            y_pred = model(X, subject_idxs, sensor_locs)
            loss = F.cross_entropy(y_pred, y)
            train_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            acc = accuracy(y_pred, y)
            train_acc.append(acc.item())
            torch.cuda.empty_cache()  # ミニバッチごとにメモリを解放

        model.eval()
        for X, y, subject_idxs in val_loader:
            X, y, subject_idxs = X.to(device), y.to(device), subject_idxs.to(device)
            with torch.no_grad():
                y_pred = model(X, subject_idxs)
            val_loss.append(F.cross_entropy(y_pred, y).item())
            val_acc.append(accuracy(y_pred, y).item())
            torch.cuda.empty_cache()  # ミニバッチごとにメモリを解放

        if rank == 0:
            print(f"Epoch {epoch+1}/{cfg.epochs} | train loss: {np.mean(train_loss):.3f} | train acc: {np.mean(train_acc):.3f} | val loss: {np.mean(val_loss):.3f} | val acc: {np.mean(val_acc):.3f}")
            torch.save(model.state_dict(), os.path.join(logdir, "model_last.pt"))
            if cfg.use_wandb:
                wandb.log({"train_loss": np.mean(train_loss), "train_acc": np.mean(train_acc), "val_loss": np.mean(val_loss), "val_acc": np.mean(val_acc)})
        
        mean_val_acc = np.mean(val_acc)
        if mean_val_acc > max_val_acc:
            if rank == 0:
                cprint("New best.", "cyan")
                torch.save(model.state_dict(), os.path.join(logdir, "model_best.pt"))
            max_val_acc = mean_val_acc
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
        
        if early_stopping_counter >= early_stopping_patience:
            if rank == 0:
                cprint("Early stopping triggered.", "red")
            break
        
        cosine_scheduler.step()
        step_scheduler.step()

        # 10エポックごとに明示的にメモリを解放
        if (epoch + 1) % 10 == 0:
            torch.cuda.empty_cache()
    
    # ----------------------------------
    #  Start evaluation with best model
    # ----------------------------------
    if rank == 0:
        model.load_state_dict(torch.load(os.path.join(logdir, "model_best.pt"), map_location=device))

    dist.barrier()  # 各プロセスが評価を始める前に同期します

    preds = [] 
    model.eval()
    for X, subject_idxs in test_loader:
        X, subject_idxs = X.to(device), subject_idxs.to(device)
        
        with torch.no_grad():
            pred = model(X, subject_idxs)
            preds.append(pred.detach().cpu())
        
        # ミニバッチごとにメモリを解放
        torch.cuda.empty_cache()

    if rank == 0:
        preds = torch.cat(preds, dim=0).numpy()
        np.save(os.path.join(logdir, "submission.npy"), preds)
        cprint(f"Submission {preds.shape} saved at {logdir}", "cyan")

    cleanup()



@hydra.main(version_base=None, config_path="configs", config_name="MEG_config")
def main(cfg: DictConfig): 
    
    world_size = 1  # 使用可能なGPU数を取得
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)  # Hydraの設定を辞書に変換

    # Hydra設定をファイルに保存
    hydra_cfg = HydraConfig.get()
    hydra_run_dir = hydra_cfg.run.dir
    with open("hydra_config.json", "w") as f:
        json.dump({"hydra_run_dir": hydra_run_dir}, f)
    
    mp.spawn(run, args=(world_size, cfg_dict, hydra_run_dir), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()