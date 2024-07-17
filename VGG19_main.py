import sys
import os
import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics import Accuracy
import hydra
from omegaconf import DictConfig
import wandb
from termcolor import cprint
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.models import vgg19_bn
import gc

from src.VGG19_datasets import ThingsMEGDataset_VGG19
from src.VGG19_utils import set_seed

# collectionsモジュールを使わずにエラーを回避するためのパッチ
import collections
collections.Container = collections.abc.Collection

def free_gpu_memory():
    torch.cuda.empty_cache()
    gc.collect()

@hydra.main(version_base=None, config_path="configs", config_name="VGG19_config")
def run(args: DictConfig):
    set_seed(args.seed)
    logdir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    
    if args.use_wandb:
        wandb.init(mode="online", dir=logdir, project="MEG-classification")

    # ------------------
    #    Dataloader
    # ------------------
    loader_args = {"batch_size": args.batch_size, "num_workers": args.num_workers}
    
    train_set = ThingsMEGDataset_VGG19("train", args.data_dir)
    train_loader = torch.utils.data.DataLoader(train_set, shuffle=True, **loader_args)
    val_set = ThingsMEGDataset_VGG19("val", args.data_dir)
    val_loader = torch.utils.data.DataLoader(val_set, shuffle=False, **loader_args)
    test_set = ThingsMEGDataset_VGG19("test", args.data_dir)
    test_loader = torch.utils.data.DataLoader(
        test_set, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers
    )

    # ------------------
    #       Model
    # ------------------
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = vgg19_bn(pretrained=True)
    model.classifier = torch.nn.Sequential(
        torch.nn.Linear(25088, 4096),
        torch.nn.ReLU(True),
        torch.nn.Dropout(p=0.6),
        torch.nn.Linear(4096, 4096),
        torch.nn.ReLU(True),
        torch.nn.Dropout(p=0.6),
        torch.nn.Linear(4096, train_set.num_classes),
    )
    
    if torch.cuda.device_count() > 1:
        print("Using GPUs: 0 and 1")
        model = torch.nn.DataParallel(model, device_ids=[0, 1])  # GPU0とGPU1を使用
    else:
        print("Using a single GPU")

    model = model.to(device)

    # ------------------
    #     Optimizer
    # ------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=8, verbose=True)

    # ------------------
    #   Early Stopping
    # ------------------
    early_stopping_patience = 20
    early_stopping_counter = 0
    max_val_acc = 0

    # ------------------
    #   Start training
    # ------------------  
    accuracy = Accuracy(
        task="multiclass", num_classes=train_set.num_classes, top_k=10
    ).to(device)
      
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        
        train_loss, train_acc, val_loss, val_acc = [], [], [], []
        
        model.train()
        for X, y, subject_idxs in train_loader:
            X, y = X.to(device), y.to(device)

            y_pred = model(X)
            
            loss = F.cross_entropy(y_pred, y)
            train_loss.append(loss.item())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            acc = accuracy(y_pred, y)
            train_acc.append(acc.item())

        model.eval()
        for X, y, subject_idxs in val_loader:
            X, y = X.to(device), y.to(device)
            
            with torch.no_grad():
                y_pred = model(X)
            
            val_loss.append(F.cross_entropy(y_pred, y).item())
            val_acc.append(accuracy(y_pred, y).item())

        # Log metrics and update scheduler
        avg_train_loss = np.mean(train_loss)
        avg_val_loss = np.mean(val_loss)
        avg_train_acc = np.mean(train_acc)
        avg_val_acc = np.mean(val_acc)

        print(f"Epoch {epoch+1}/{args.epochs} | train loss: {avg_train_loss:.3f} | train acc: {avg_train_acc:.3f} | val loss: {avg_val_loss:.3f} | val acc: {avg_val_acc:.3f}")

        torch.save(model.state_dict(), os.path.join(logdir, "model_last.pt"))
        if args.use_wandb:
            wandb.log({"train_loss": avg_train_loss, "train_acc": avg_train_acc, "val_loss": avg_val_loss, "val_acc": avg_val_acc})
        
        if avg_val_acc > max_val_acc:
            cprint("New best.", "cyan")
            torch.save(model.state_dict(), os.path.join(logdir, "model_best.pt"))
            max_val_acc = avg_val_acc
            early_stopping_counter = 0  # リセット
        else:
            early_stopping_counter += 1
        
        # Update the learning rate scheduler
        scheduler.step(avg_val_acc)

        # Check early stopping
        if early_stopping_counter >= early_stopping_patience:
            cprint("Early stopping triggered.", "red")
            break

        # 10エポックごとにGPUメモリをクリア
        if (epoch + 1) % 10 == 0:
            free_gpu_memory()
    
    print("Training complete, starting evaluation")

    # ----------------------------------
    #  Start evaluation with best model
    # ----------------------------------
    model.load_state_dict(torch.load(os.path.join(logdir, "model_best.pt"), map_location=device))
    print("Loaded best model for evaluation")

    preds = [] 
    model.eval()
    for X, subject_idxs in test_loader:        
        preds.append(model(X.to(device)).detach().cpu())
        
    preds = torch.cat(preds, dim=0).numpy()
    np.save(os.path.join(logdir, "submission"), preds)
    cprint(f"Submission {preds.shape} saved at {logdir}", "cyan")

if __name__ == "__main__":
    run()
