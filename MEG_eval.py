import os
import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics import Accuracy
import hydra
from omegaconf import DictConfig
import wandb
from termcolor import cprint
from tqdm import tqdm

from src.datasets import ThingsMEGDataset
from src.models import LSTMConvClassifier
from src.utils import set_seed

@torch.no_grad()
@hydra.main(version_base=None, config_path="configs", config_name="config")
def run(args: DictConfig):
    set_seed(args.seed)
    logdir = os.path.dirname(args.model_path)
    
    if args.use_wandb:
        wandb.init(mode="online", dir=logdir, project="MEG-classification")
    
    # ------------------
    #    Dataloader
    # ------------------    
    test_set = ThingsMEGDataset("test", args.data_dir)
    test_loader = torch.utils.data.DataLoader(
        test_set, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers
    )

    # ------------------
    #       Model
    # ------------------
    model = LSTMConvClassifier(
        num_classes=test_set.num_classes,
        seq_len=test_set.seq_len,
        in_channels=test_set.num_channels,
        hid_dim=args.hid_dim,
        num_blocks=args.num_blocks,
        kernel_size=args.kernel_size,
        lstm_hidden_dim=args.lstm_hidden_dim,
        lstm_layers=args.lstm_layers,
        dropout_prob=args.dropout_prob
    ).to(args.device)
    model.load_state_dict(torch.load(args.model_path, map_location=args.device))

    # ------------------
    #  Start evaluation
    # ------------------ 
    preds = []
    labels = []
    model.eval()
    for X, y, subject_idxs in tqdm(test_loader, desc="Evaluation"):
        y_pred = model(X.to(args.device), subject_idxs.to(args.device)).detach().cpu()
        preds.append(y_pred)
        labels.append(y.cpu())
        
    preds = torch.cat(preds, dim=0)
    labels = torch.cat(labels, dim=0)
    accuracy = Accuracy(num_classes=test_set.num_classes)
    acc = accuracy(preds, labels)

    print(f"Test accuracy: {acc:.4f}")
    if args.use_wandb:
        wandb.log({"test_accuracy": acc})

    # Save predictions
    preds = preds.numpy()
    labels = labels.numpy()
    np.save(os.path.join(logdir, "preds.npy"), preds)
    np.save(os.path.join(logdir, "labels.npy"), labels)
    cprint(f"Predictions and labels saved at {logdir}", "cyan")


if __name__ == "__main__":
    run()
