import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
import hydra
from omegaconf import DictConfig
import wandb
from termcolor import cprint
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torchvision import transforms

from src.MEG_datasets import ThingsMEGDataset, ImageDataset
from src.MEG_models import fclip
from src.MEG_utils import set_seed

@hydra.main(version_base=None, config_path="configs", config_name="MEG_config")
def run(args: DictConfig):
    set_seed(args.seed)
    logdir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    
    if args.use_wandb:
        wandb.init(mode="online", dir=logdir, project="MEG-classification")
    
    # ------------------
    #    Dataloader
    # ------------------
    loader_args = {"batch_size": args.batch_size, "num_workers": args.num_workers}
    print("Debug start")
        
    train_set = ThingsMEGDataset("train", args.data_dir)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    batches = [batch for batch in train_loader]

    train_loader = np.stack(batches)
        
    if train_loader.ndim == 3 and train_loader.shape[1] > train_loader.shape[2]:
        # 形状が [バッチサイズ, センサー数, 時間サンプル] の場合、調整不要
        pass
    elif train_loader.ndim == 3 and train_loader.shape[1] < train_loader.shape[2]:
        # 形状が [バッチサイズ, 時間サンプル, センサー数] の場合、 [バッチサイズ, センサー数, 時間サンプル] に変換
        train_loader = train_loader.transpose(0, 2, 1)

    # データをテンソルに変換
    train_loader = torch.tensor(train_loader, dtype=torch.float32)

    print("Train load complete")
    
    val_set = ThingsMEGDataset("val", args.data_dir)
    val_loader = DataLoader(val_set, shuffle=False, **loader_args)
   
    print("val load complete")

    test_set = ThingsMEGDataset("test", args.data_dir)
    test_loader = DataLoader(test_set, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers)
    
    print("test load complete")

    # 画像データセット
    transform = transforms.Compose([transforms.ToTensor()])
    image_train_set = ImageDataset(split='train', images_dir='Images', data_dir=args.data_dir, transform=transform)
    image_train_loader = DataLoader(image_train_set, shuffle=True, **loader_args)

    image_val_set = ImageDataset(split='val', images_dir='Images', data_dir=args.data_dir, transform=transform)
    image_val_loader = DataLoader(image_val_set, shuffle=False, **loader_args)

    image_test_set = ImageDataset(split='test', images_dir='Images', data_dir=args.data_dir, transform=transform)
    image_test_loader = DataLoader(image_test_set, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers)

    # ------------------
    #       Model
    # ------------------
    num_sensors = 306  # 仮のセンサー数
    sensor_locs = np.random.rand(num_sensors, 2)  # 2D位置
    sensor_locs = torch.tensor(sensor_locs, dtype=torch.float32)

    # 被験者IDの設定
    subject_ids = np.random.randint(0, 4, size=args.batch_size)  # 0から3までの被験者IDをランダムに生成
    subject_ids = torch.tensor(subject_ids, dtype=torch.long)

    # fclipモデルを初期化
    num_input_channels = 306  # 例として仮の入力チャネル数
    num_output_channels = 128  # 任意の出力チャネル数
    num_subjects = 4  # 仮の被験者数
    num_classes = 1854  # クラス数（画像カテゴリ数）
    model = fclip(num_input_channels, num_output_channels, num_subjects, num_classes)

    # デバイス設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs.")
        model = torch.nn.DataParallel(model, device_ids=[0, 1])
    model = model.to(device)


    # ------------------
    #     Optimizer
    # ------------------
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    step_scheduler = StepLR(optimizer, step_size=10, gamma=0.5)  # 10エポックごとに学習率を半減

    # ------------------
    #   Start training
    # ------------------  
    max_val_acc = 0
    accuracy = Accuracy(
        task="multiclass", num_classes=train_set.num_classes, top_k=10
    ).to(args.device)
    
    early_stopping_patience = 30
    early_stopping_counter = 0
    
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        
        train_loss, train_acc, val_loss, val_acc = [], [], [], []
        
        model.train()
        for X, y, subject_idxs in train_loader:
            X, y, subject_idxs = X.to(args.device), y.to(args.device), subject_idxs.to(args.device)

            y_pred = model(X, subject_idxs, sensor_locs)
            
            loss = F.cross_entropy(y_pred, y)
            train_loss.append(loss.item())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            acc = accuracy(y_pred, y)
            train_acc.append(acc.item())

        model.eval()
        for X, y, subject_idxs in val_loader:
            X, y, subject_idxs = X.to(args.device), y.to(args.device), subject_idxs.to(args.device)
            
            with torch.no_grad():
                y_pred = model(X, subject_idxs)
            
            val_loss.append(F.cross_entropy(y_pred, y).item())
            val_acc.append(accuracy(y_pred, y).item())

        print(f"Epoch {epoch+1}/{args.epochs} | train loss: {np.mean(train_loss):.3f} | train acc: {np.mean(train_acc):.3f} | val loss: {np.mean(val_loss):.3f} | val acc: {np.mean(val_acc):.3f}")
        torch.save(model.state_dict(), os.path.join(logdir, "model_last.pt"))
        if args.use_wandb:
            wandb.log({"train_loss": np.mean(train_loss), "train_acc": np.mean(train_acc), "val_loss": np.mean(val_loss), "val_acc": np.mean(val_acc)})
        
        mean_val_acc = np.mean(val_acc)
        if mean_val_acc > max_val_acc:
            cprint("New best.", "cyan")
            torch.save(model.state_dict(), os.path.join(logdir, "model_best.pt"))
            max_val_acc = mean_val_acc
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
        
        if early_stopping_counter >= early_stopping_patience:
            cprint("Early stopping triggered.", "red")
            break
        
        cosine_scheduler.step()
        step_scheduler.step()
    
    # ----------------------------------
    #  Start evaluation with best model
    # ----------------------------------
    model.load_state_dict(torch.load(os.path.join(logdir, "model_best.pt"), map_location=args.device))

    preds = [] 
    model.eval()
    for X, subject_idxs in test_loader:        
        preds.append(model(X.to(args.device), subject_idxs.to(args.device)).detach().cpu())
        
    preds = torch.cat(preds, dim=0).numpy()
    np.save(os.path.join(logdir, "submission"), preds)
    cprint(f"Submission {preds.shape} saved at {logdir}", "cyan")


if __name__ == "__main__":
    run()

    print("Train load complete")
    
    val_set = ThingsMEGDataset("val", args.data_dir)
    val_loader = DataLoader(val_set, shuffle=False, **loader_args)
    print("val load complete")

    test_set = ThingsMEGDataset("test", args.data_dir)
    test_loader = DataLoader(test_set, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers)
    print("test load complete")

    # 画像データセット
    transform = transforms.Compose([transforms.ToTensor()])
    image_train_set = ImageDataset(split='train', images_dir='Images', data_dir=args.data_dir, transform=transform)
    image_train_loader = DataLoader(image_train_set, shuffle=True, **loader_args)

    image_val_set = ImageDataset(split='val', images_dir='Images', data_dir=args.data_dir, transform=transform)
    image_val_loader = DataLoader(image_val_set, shuffle=False, **loader_args)

    image_test_set = ImageDataset(split='test', images_dir='Images', data_dir=args.data_dir, transform=transform)
    image_test_loader = DataLoader(image_test_set, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers)

    # ------------------
    #       Model
    # ------------------
    model = fclip(
        num_classes=train_set.num_classes,
        seq_len=train_set.seq_len,
        in_channels=train_set.num_channels,
        hid_dim=args.hid_dim,
        num_blocks=args.num_blocks,
        kernel_size=args.kernel_size,
        lstm_hidden_dim=args.lstm_hidden_dim,
        lstm_layers=args.lstm_layers,
        dropout_prob=args.dropout_prob
    ).to(args.device)

    # ------------------
    #     Optimizer
    # ------------------
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    step_scheduler = StepLR(optimizer, step_size=10, gamma=0.5)  # 10エポックごとに学習率を半減

    # ------------------
    #   Start training
    # ------------------  
    max_val_acc = 0
    accuracy = Accuracy(
        task="multiclass", num_classes=train_set.num_classes, top_k=10
    ).to(args.device)
    
    early_stopping_patience = 30
    early_stopping_counter = 0
      
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        
        train_loss, train_acc, val_loss, val_acc = [], [], [], []
        
        model.train()
        for X, y, subject_idxs in train_loader:
            X, y, subject_idxs = X.to(args.device), y.to(args.device), subject_idxs.to(args.device)

            y_pred = model(X, subject_idxs)
            
            loss = F.cross_entropy(y_pred, y)
            train_loss.append(loss.item())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            acc = accuracy(y_pred, y)
            train_acc.append(acc.item())

        model.eval()
        for X, y, subject_idxs in val_loader:
            X, y, subject_idxs = X.to(args.device), y.to(args.device), subject_idxs.to(args.device)
            
            with torch.no_grad():
                y_pred = model(X, subject_idxs)
            
            val_loss.append(F.cross_entropy(y_pred, y).item())
            val_acc.append(accuracy(y_pred, y).item())

        print(f"Epoch {epoch+1}/{args.epochs} | train loss: {np.mean(train_loss):.3f} | train acc: {np.mean(train_acc):.3f} | val loss: {np.mean(val_loss):.3f} | val acc: {np.mean(val_acc):.3f}")
        torch.save(model.state_dict(), os.path.join(logdir, "model_last.pt"))
        if args.use_wandb:
            wandb.log({"train_loss": np.mean(train_loss), "train_acc": np.mean(train_acc), "val_loss": np.mean(val_loss), "val_acc": np.mean(val_acc)})
        
        mean_val_acc = np.mean(val_acc)
        if mean_val_acc > max_val_acc:
            cprint("New best.", "cyan")
            torch.save(model.state_dict(), os.path.join(logdir, "model_best.pt"))
            max_val_acc = mean_val_acc
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
        
        if early_stopping_counter >= early_stopping_patience:
            cprint("Early stopping triggered.", "red")
            break
        
        cosine_scheduler.step()
        step_scheduler.step()
    
    # ----------------------------------
    #  Start evaluation with best model
    # ----------------------------------
    model.load_state_dict(torch.load(os.path.join(logdir, "model_best.pt"), map_location=args.device))

    preds = [] 
    model.eval()
    for X, subject_idxs in test_loader:        
        preds.append(model(X.to(args.device), subject_idxs.to(args.device)).detach().cpu())
        
    preds = torch.cat(preds, dim=0).numpy()
    np.save(os.path.join(logdir, "submission"), preds)
    cprint(f"Submission {preds.shape} saved at {logdir}", "cyan")


if __name__ == "__main__":
    run()
