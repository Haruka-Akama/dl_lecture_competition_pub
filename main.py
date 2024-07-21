import os
import numpy as np
import torch
import json
import torch.nn.functional as F
from torchmetrics import Accuracy
import hydra
from omegaconf import DictConfig
from omegaconf import DictConfig, OmegaConf
import wandb
from termcolor import cprint
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from collections.abc import Container  # 修正点: collections.Container -> collections.abc.Container
from hydra.core.hydra_config import HydraConfig
#from src.datasets_preprocess import ThingsMEGDataset
from src.datasets import ThingsMEGDataset
from src.models import LSTMConvClassifier
from src.utils import set_seed

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
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
    loader_args = {"batch_size": cfg.batch_size, "num_workers": cfg.num_workers}
    print(f"Debug start on rank {rank}")
        
    train_set = ThingsMEGDataset("train", cfg.data_dir)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set, num_replicas=world_size, rank=rank)
    train_loader = torch.utils.data.DataLoader(train_set, sampler=train_sampler, **loader_args)
    print(f"Train load complete on rank {rank}")
    
    val_set = ThingsMEGDataset("val", cfg.data_dir)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_set, num_replicas=world_size, rank=rank)
    val_loader = torch.utils.data.DataLoader(val_set, sampler=val_sampler, **loader_args)
    print(f"Val load complete on rank {rank}")

    test_set = ThingsMEGDataset("test", cfg.data_dir)
    test_loader = torch.utils.data.DataLoader(test_set, **loader_args)
    print(f"Test load complete on rank {rank}")

    # ------------------
    #       Model
    # ------------------
    model = LSTMConvClassifier(
        train_set.num_classes, train_set.seq_len, train_set.num_channels, dropout_prob=0.7
    ).to(rank)

    model = DDP(model, device_ids=[rank])

    # LSTMモジュールのパラメータをフラット化して、メモリの連続性を確保
    for module in model.modules():   # LSTMモジュールのパラメータをフラット化して、メモリの連続性を確保
        if isinstance(module, torch.nn.LSTM):
                module.flatten_parameters()

    # ------------------
    #     Optimizer
    # ------------------
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    # ------------------
    #   Start training
    # ------------------  
    max_val_acc = 0
    accuracy = Accuracy(
        task="multiclass", num_classes=train_set.num_classes, top_k=10
    ).to(rank)
    
    early_stopping_patience = 10
    early_stopping_counter = 0
      
    for epoch in range(cfg.epochs):
        print(f"Epoch {epoch+1}/{cfg.epochs} on rank {rank}")
        
        train_loss, train_acc, val_loss, val_acc = [], [], [], []
        
        model.train()
        for batch_idx, (X, y, subject_idxs) in enumerate(train_loader):
            X, y, subject_idxs = X.to(rank), y.to(rank), subject_idxs.to(rank)

            y_pred = model(X, subject_idxs)
            
            loss = F.cross_entropy(y_pred, y)
            train_loss.append(loss.item())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            acc = accuracy(y_pred, y)
            train_acc.append(acc.item())

            if (batch_idx + 1) % 10 == 0 and rank == 0:
                print(f"Train Epoch: {epoch+1} [{batch_idx * len(X)}/{len(train_loader.dataset)}] Loss: {loss.item():.6f} Accuracy: {acc.item():.6f}")

        model.eval()
        for batch_idx, (X, y, subject_idxs) in enumerate(val_loader):
            X, y, subject_idxs = X.to(rank), y.to(rank), subject_idxs.to(rank)
            
            with torch.no_grad():
                y_pred = model(X, subject_idxs)
            
            val_loss.append(F.cross_entropy(y_pred, y).item())
            val_acc.append(accuracy(y_pred, y).item())

            if (batch_idx + 1) % 10 == 0 and rank == 0:
                print(f"Validation Epoch: {epoch+1} [{batch_idx * len(X)}/{len(val_loader.dataset)}] Loss: {val_loss[-1]:.6f} Accuracy: {val_acc[-1]:.6f}")

        if rank == 0:
            print(f"Epoch {epoch+1}/{cfg.epochs} | train loss: {np.mean(train_loss):.3f} | train acc: {np.mean(train_acc):.3f} | val loss: {np.mean(val_loss):.3f} | val acc: {np.mean(val_acc):.3f}")
            torch.save(model.state_dict(), os.path.join(logdir, "model_last.pt"))
            if cfg.use_wandb:
                wandb.log({"train_loss": np.mean(train_loss), "train_acc": np.mean(train_acc), "val_loss": np.mean(val_loss), "val_acc": np.mean(val_acc)})
        
        mean_val_acc = np.mean(val_acc)
        if mean_val_acc > max_val_acc and rank == 0:
            cprint("New best.", "cyan")
            torch.save(model.state_dict(), os.path.join(logdir, "model_best.pt"))
            max_val_acc = mean_val_acc
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
        
        if early_stopping_counter >= early_stopping_patience:
            cprint("Early stopping triggered.", "red")
            break
        
        scheduler.step()
    
    if rank == 0:
        # ----------------------------------
        #  Start evaluation with best model
        # ----------------------------------
        model.load_state_dict(torch.load(os.path.join(logdir, "model_best.pt"), map_location=args.device))

        preds = [] 
        model.eval()
        for X, subject_idxs in test_loader:        
            preds.append(model(X.to(rank), subject_idxs.to(rank)).detach().cpu())
            
        preds = torch.cat(preds, dim=0).numpy()
        np.save(os.path.join(logdir, "submission"), preds)
        cprint(f"Submission {preds.shape} saved at {logdir}", "cyan")
    
    cleanup()

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig): 
    
    world_size = 2
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)  # Hydraの設定を辞書に変換

    # Hydra設定をファイルに保存
    hydra_cfg = HydraConfig.get()
    hydra_run_dir = hydra_cfg.run.dir
    with open("hydra_config.json", "w") as f:
        json.dump({"hydra_run_dir": hydra_run_dir}, f)
    
    mp.spawn(run, args=(world_size, cfg_dict, hydra_run_dir), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()
