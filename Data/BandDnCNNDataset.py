import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class BandDnCNNDataset(Dataset):
    """
    Dataset for DnCNN using pre-saved grayscale band .npy files.

    Expected folder structure:
        processed_data/
            train/
                xxx_band_000.npy
                xxx_band_001.npy
                ...
            val/
                ...
            test/
                ...

    Each .npy file should contain one 2D grayscale image: shape (H, W).

    This dataset:
    - loads one saved band file
    - optionally extracts a patch
    - adds Gaussian noise on-the-fly
    - returns tensors shaped [1, H, W]

    Args:
        data_dir (str): Folder containing .npy band files.
        sigma (float): Noise std in [0,255] scale.
        normalize (bool): Whether data is expected in [0,1].
        training (bool): Random noise and random patch if True.
        seed (int): Seed for deterministic behavior in eval mode.
        patch_size (int or None): Patch size. If None, use full image.
    """

    def __init__(
        self,
        data_dir: str,
        sigma: float = 25.0,
        normalize: bool = True,
        training: bool = True,
        seed: int = 42,
        patch_size: int | None = 64,
    ) -> None:
        super().__init__()

        self.data_dir = data_dir
        self.normalize = normalize
        self.training = training
        self.seed = seed
        self.patch_size = patch_size
        self.sigma = sigma / 255.0 if normalize else sigma

        if not os.path.isdir(self.data_dir):
            raise FileNotFoundError(f"Directory not found: {self.data_dir}")

        self.file_paths = sorted(
            [
                os.path.join(self.data_dir, name)
                for name in os.listdir(self.data_dir)
                if name.endswith(".npy")
            ]
        )

        if len(self.file_paths) == 0:
            raise ValueError(f"No .npy files found in directory: {self.data_dir}")

    def __len__(self) -> int:
        return len(self.file_paths)

    def _extract_patch(self, band: np.ndarray, index: int) -> np.ndarray:
        if self.patch_size is None:
            return band

        h, w = band.shape
        ps = self.patch_size

        if ps > h or ps > w:
            raise ValueError(f"patch_size={ps} is larger than band shape {(h, w)}")

        if self.training:
            top = np.random.randint(0, h - ps + 1)
            left = np.random.randint(0, w - ps + 1)
        else:
            rng = np.random.default_rng(self.seed + index)
            top = rng.integers(0, h - ps + 1)
            left = rng.integers(0, w - ps + 1)

        return band[top:top + ps, left:left + ps]

    def _generate_noise(self, shape, index: int) -> np.ndarray:
        if self.training:
            noise = np.random.normal(0.0, self.sigma, size=shape).astype(np.float32)
        else:
            rng = np.random.default_rng(self.seed + index)
            noise = rng.normal(0.0, self.sigma, size=shape).astype(np.float32)
        return noise

    def __getitem__(self, index: int):
        file_path = self.file_paths[index]
        clean = np.load(file_path).astype(np.float32)

        if clean.ndim != 2:
            raise ValueError(f"Expected 2D band image, got shape {clean.shape} for {file_path}")

        clean = self._extract_patch(clean, index)
        noise = self._generate_noise(clean.shape, index)
        noisy = clean + noise

        if self.normalize:
            noisy = np.clip(noisy, 0.0, 1.0)

        clean = np.expand_dims(clean, axis=0)   # [1, H, W]
        noise = np.expand_dims(noise, axis=0)
        noisy = np.expand_dims(noisy, axis=0)

        return {
            "noisy": torch.from_numpy(noisy.astype(np.float32)),
            "noise": torch.from_numpy(noise.astype(np.float32)),
            "clean": torch.from_numpy(clean.astype(np.float32)),
            "key": Path(file_path).stem,
            "file_path": file_path,
        }