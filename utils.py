import os
import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Optional deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path: str) -> None:
    """
    Create directory if it does not exist.
    """
    os.makedirs(path, exist_ok=True)


def save_checkpoint(state: dict, save_path: str) -> None:
    """
    Save checkpoint dictionary.
    """
    directory = os.path.dirname(save_path)
    if directory:
        ensure_dir(directory)
    torch.save(state, save_path)


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    device: str = "cpu",
) -> dict:
    """
    Load checkpoint into model and optionally optimizer/scheduler.

    Returns:
        checkpoint dictionary
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint


def calculate_mse(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """
    Compute mean squared error between two tensors.
    """
    return torch.mean((img1 - img2) ** 2).item()


def calculate_psnr(
    img1: torch.Tensor,
    img2: torch.Tensor,
    max_value: float = 1.0,
) -> float:
    """
    Compute PSNR between two tensors.
    """
    mse = calculate_mse(img1, img2)
    if mse == 0:
        return float("inf")
    return 10 * np.log10((max_value ** 2) / mse)


class AverageMeter:
    """
    Track and update average values such as loss or PSNR.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.sum = 0.0
        self.count = 0
        self.avg = 0.0

    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0.0


def tensor_to_numpy_image(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert tensor [C, H, W] to NumPy array.

    Returns:
        [H, W] for grayscale
        [H, W, C] for multi-channel
    """
    array = tensor.detach().cpu().numpy()

    if array.ndim != 3:
        raise ValueError(f"Expected tensor with shape [C, H, W], got {array.shape}")

    if array.shape[0] == 1:
        return array[0]
    return np.transpose(array, (1, 2, 0))


def save_numpy_image(tensor: torch.Tensor, save_path: str) -> None:
    """
    Save tensor image as .npy file.
    """
    directory = os.path.dirname(save_path)
    if directory:
        ensure_dir(directory)

    array = tensor_to_numpy_image(tensor)
    np.save(save_path, array)


def get_device() -> torch.device:
    """
    Get available device.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")