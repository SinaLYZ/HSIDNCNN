import os
import re
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class BandDnCNNDatasetSigmaMap(Dataset):
    """
    Dataset for sigma-conditioned DnCNN using pre-saved grayscale band .npy files.

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
    - finds that band's sigma from band_sigmas
    - adds Gaussian noise on-the-fly using that band's sigma
    - creates a sigma map
    - returns:
        noisy -> [2, H, W] = [noisy_band, sigma_map]
        noise -> [1, H, W]
        clean -> [1, H, W]

    Args:
        data_dir (str): Folder containing .npy band files.
        band_sigmas (dict[int, float]): Mapping like {0: 25.0, 1: 30.0, ...}
            where keys are band indices parsed from filenames such as xxx_band_000.npy
        normalize (bool): Whether data is expected in [0,1].
        training (bool): Random noise and random patch if True.
        seed (int): Seed for deterministic behavior in eval mode.
        patch_size (int or None): Patch size. If None, use full image.
    """

    def __init__(
        self,
        data_dir: str,
        band_sigmas: dict[int, float],
        normalize: bool = True,
        training: bool = True,
        seed: int = 42,
        patch_size: int | None = 64,
    ) -> None:
        super().__init__()

        self.data_dir = data_dir
        self.band_sigmas = band_sigmas
        self.normalize = normalize
        self.training = training
        self.seed = seed
        self.patch_size = patch_size

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

    def _parse_band_index(self, file_path: str) -> int:
        stem = Path(file_path).stem

        # Expected match: xxx_band_000
        match = re.search(r"_band_(\d+)$", stem)
        if match is None:
            raise ValueError(
                f"Could not parse band index from filename: {file_path}\n"
                f"Expected format like xxx_band_000.npy"
            )

        return int(match.group(1))

    def _generate_noise(self, shape, sigma_value: float, index: int) -> np.ndarray:
        if self.training:
            noise = np.random.normal(0.0, sigma_value, size=shape).astype(np.float32)
        else:
            rng = np.random.default_rng(self.seed + index)
            noise = rng.normal(0.0, sigma_value, size=shape).astype(np.float32)
        return noise

    def __getitem__(self, index: int):
        file_path = self.file_paths[index]
        clean = np.load(file_path).astype(np.float32)

        if clean.ndim != 2:
            raise ValueError(f"Expected 2D band image, got shape {clean.shape} for {file_path}")

        clean = self._extract_patch(clean, index)

        band_idx = self._parse_band_index(file_path)

        if band_idx not in self.band_sigmas:
            raise KeyError(
                f"No sigma found for band index {band_idx}. "
                f"Make sure band_sigmas contains this band."
            )

        sigma_raw = self.band_sigmas[band_idx]
        sigma_value = sigma_raw / 255.0 if self.normalize else sigma_raw

        noise = self._generate_noise(clean.shape, sigma_value, index)
        noisy = clean + noise

        if self.normalize:
            noisy = np.clip(noisy, 0.0, 1.0)

        sigma_map = np.full_like(clean, sigma_value, dtype=np.float32)

        clean = np.expand_dims(clean, axis=0)          # [1, H, W]
        noise = np.expand_dims(noise, axis=0)          # [1, H, W]
        noisy = np.expand_dims(noisy, axis=0)          # [1, H, W]
        sigma_map = np.expand_dims(sigma_map, axis=0)  # [1, H, W]

        model_input = np.concatenate([noisy, sigma_map], axis=0)  # [2, H, W]

        return {
            "noisy": torch.from_numpy(model_input.astype(np.float32)),
            "noise": torch.from_numpy(noise.astype(np.float32)),
            "clean": torch.from_numpy(clean.astype(np.float32)),
            "sigma": torch.tensor([sigma_value], dtype=torch.float32),
            "band_idx": band_idx,
            "key": Path(file_path).stem,
            "file_path": file_path,
        }