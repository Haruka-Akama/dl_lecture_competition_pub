import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
import hydra
from omegaconf import DictConfig
from termcolor import cprint
from tqdm import tqdm

from MEG_datasets import ThingsMEGDataset, ImageDataset
from MEG_models import fclip
from MEG_utils import set_seed

@torch.no_grad()
@hydra.main(version_base=None, config_path="configs", config_name="MEG_config")
def run(args: DictConfig):
    set_seed(args.seed)
    savedir = os.path.dirname(args.model_path)
    
    # ------------------
    #    Dataloader
    # ------------------    
    test_set = ThingsMEGDataset("test", args.data_dir)
    test_loader = DataLoader(test_set, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers)

    # 画像データセット
    transform = transforms.Compose([transforms.ToTensor()])
    image_test_set = ImageDataset(split='test', images_dir='workspace/dl_lecture_competition_pub/data/Images'
    image_test_set = ImageDataset(split='test', images_dir=args.images_dir, data_dir=args.data_dir, transform=transform)
    image_test_loader = DataLoader(image_test_set, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers)

    # ------------------
    #       Model
    # ------------------
    model = fclip(
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
    model.eval()
    for X, subject_idxs in tqdm(test_loader, desc="Validation"):        
        preds.append(model(X.to(args.device), subject_idxs.to(args.device)).detach().cpu())
        
    preds = torch.cat(preds, dim=0).numpy()
    np.save(os.path.join(savedir, "submission"), preds)
    cprint(f"Submission {preds.shape} saved at {savedir}", "cyan")

    # 画像データの評価
    image_preds = [] 
    for images, subject_idxs in tqdm(image_test_loader, desc="Image Validation"):
        images = images.to(args.device)
        subject_idxs = subject_idxs.to(args.device)
        outputs = model(images, subject_idxs)
        image_preds.append(outputs.detach().cpu())
        
    image_preds = torch.cat(image_preds, dim=0).numpy()
    np.save(os.path.join(savedir, "image_submission"), image_preds)
    cprint(f"Image submission {image_preds.shape} saved at {savedir}", "cyan")


if __name__ == "__main__":
    run()
