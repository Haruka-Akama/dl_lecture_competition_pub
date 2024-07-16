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
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR

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
        for X, y, subject_idxs in tqdm(train_loader, desc="Train"):
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
        for X, y, subject_idxs in tqdm(val_loader, desc="Validation"):
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
    for X, subject_idxs in tqdm(test_loader, desc="Validation"):        
        preds.append(model(X.to(args.device), subject_idxs.to(args.device)).detach().cpu())
        
    preds = torch.cat(preds, dim=0).numpy()
    np.save(os.path.join(logdir, "submission"), preds)
    cprint(f"Submission {preds.shape} saved at {logdir}", "cyan")


if __name__ == "__main__":
    run()
