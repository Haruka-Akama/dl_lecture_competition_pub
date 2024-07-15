import os, sys
import numpy as np
import mne
import torch
from PIL import Image
from sklearn.preprocessing import RobustScaler
from functools import partial
from termcolor import cprint
from tqdm import tqdm
from typing import Tuple, List
import hydra
from omegaconf import DictConfig

from transformers import AutoProcessor, CLIPVisionModel
import clip

from image_decoding.utils import sequential_apply


def scale_clamp(
    X: np.ndarray,
    clamp_lim: float = 5.0,
    clamp: bool = True,
    scale_transposed: bool = True,
) -> np.ndarray:
    X = RobustScaler().fit_transform(X.T if scale_transposed else X)

    if scale_transposed:
        X = X.T

    if clamp:
        X = X.clip(min=-clamp_lim, max=clamp_lim)

    return X


@torch.no_grad()
def encode_images(y_list: List[str], preprocess, clip_model, device) -> torch.Tensor:
    if isinstance(clip_model, CLIPVisionModel):
        last_hidden_states = []

        for y in tqdm(y_list, desc="Preprocessing & encoding images"):
            model_input = preprocess(images=Image.open(y), return_tensors="pt")
            model_output = clip_model(**model_input.to(device))
            last_hidden_states.append(model_output.last_hidden_state.cpu())

        return torch.cat(last_hidden_states, dim=0)
    else:
        model_input = torch.stack(
            [preprocess(Image.open(y).convert("RGB")) for y in tqdm(y_list, desc="Preprocessing images")]
        )

        return sequential_apply(
            model_input,
            clip_model.encode_image,
            batch_size=32,
            device=device,
            desc="Encoding images",
        ).float()


@hydra.main(version_base=None, config_path="./", config_name="META_config")
def run(args: DictConfig) -> None:
    save_dir = os.path.join(args.save_dir, "preproc")
    os.makedirs(save_dir, exist_ok=True)

    # Paths for MEG data
    train_X_path = os.path.join(args.data_dir, "train_X.pt")
    val_X_path = os.path.join(args.data_dir, "val_X.pt")
    test_X_path = os.path.join(args.data_dir, "test_X.pt")

    train_subject_idxs_path = os.path.join(args.data_dir, "train_subject_idxs.pt")
    val_subject_idxs_path = os.path.join(args.data_dir, "val_subject_idxs.pt")
    test_subject_idxs_path = os.path.join(args.data_dir, "test_subject_idxs.pt")

    train_y_path = os.path.join(args.data_dir, "train_y.pt")
    val_y_path = os.path.join(args.data_dir, "val_y.pt")

    train_image_paths_path = os.path.join(args.data_dir, "train_image_paths.txt")
    val_image_paths_path = os.path.join(args.data_dir, "val_image_paths.txt")

    X_list = []
    subject_idxs_list = []
    y_list = []

    # Load MEG data
    if not args.skip_meg:
        train_X = torch.load(train_X_path)
        val_X = torch.load(val_X_path)
        test_X = torch.load(test_X_path)

        train_subject_idxs = torch.load(train_subject_idxs_path)
        val_subject_idxs = torch.load(val_subject_idxs_path)
        test_subject_idxs = torch.load(test_subject_idxs_path)

        train_y = torch.load(train_y_path)
        val_y = torch.load(val_y_path)

        # Combine train and val data for processing
        X_list.extend([train_X, val_X])
        subject_idxs_list.extend([train_subject_idxs, val_subject_idxs])
        y_list.extend([train_y, val_y])

        combined_X = torch.cat(X_list, dim=0)
        combined_subject_idxs = torch.cat(subject_idxs_list, dim=0)
        combined_y = torch.cat(y_list, dim=0)

        cprint(f"Combined MEG data: {combined_X.shape}", "cyan")

        # Save the combined MEG data
        torch.save(combined_X, os.path.join(save_dir, "combined_X.pt"))
        torch.save(combined_subject_idxs, os.path.join(save_dir, "combined_subject_idxs.pt"))
        torch.save(combined_y, os.path.join(save_dir, "combined_y.pt"))

    # Load and encode images
    if not args.skip_images:
        with open(train_image_paths_path, "r") as f:
            train_image_paths = f.read().splitlines()

        with open(val_image_paths_path, "r") as f:
            val_image_paths = f.read().splitlines()

        # Combine train and val image paths
        combined_image_paths = train_image_paths + val_image_paths

        if args.vision_model.startswith("ViT-"):
            clip_model, preprocess = clip.load(args.vision_model)
            clip_model = clip_model.eval().to(device)

        elif args.vision_model.startswith("openai/"):
            clip_model = CLIPVisionModel.from_pretrained(args.vision_model).to(device)
            preprocess = AutoProcessor.from_pretrained(args.vision_model)
        else:
            raise ValueError(f"Unknown pretrained CLIP type: {args.vision_model}")

        device = f"cuda:{args.cuda_id}"

        combined_Y = encode_images(combined_image_paths, preprocess, clip_model, device)

        cprint(f"Encoded images: {combined_Y.shape}", "cyan")

        torch.save(combined_Y, os.path.join(save_dir, "combined_Y.pt"))

    if args.chance:
        combined_X = combined_X[torch.randperm(len(combined_X))]
        cprint("Data shuffled for chance level", "cyan")

    del X_list, subject_idxs_list, y_list  # Clear lists to free memory
    gc.collect()


if __name__ == "__main__":
    run()
