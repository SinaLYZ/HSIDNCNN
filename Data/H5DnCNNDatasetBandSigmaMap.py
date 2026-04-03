import os
import json
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class H5DnCNNDatasetBandSigmaMap(Dataset):
    """
    H5 dataset for sigma-conditioned DnCNN using hyperspectral cubes.

    Each spectral band is treated as one grayscale image.
    This dataset automatically generates ONE fixed sigma value per band,
    then uses that sigma whenever that band is loaded.

    Returned sample:
        noisy      -> [2, H, W] = [noisy_band, sigma_map]
        sigma_map  -> [1, H, W]
        noise      -> [1, H, W]
        clean      -> [1, H, W]
        sigma      -> scalar tensor [1]
    """

    def __init__(
        self,
        h5_path: str,
        group_name: str = "hsi_27",
        sigma_min: float = 5.0,
        sigma_max: float = 50.0,
        normalize: bool = True,
        training: bool = True,
        seed: int = 42,
        patch_size: int | None = 64,
        band_indices=None,
        dataset_paths=None,
        sigma_json_path: str | None = None,
        load_sigma_json_if_exists: bool = True,
    ) -> None:
        super().__init__()

        self.h5_path = h5_path
        self.group_name = group_name
        self.normalize = normalize
        self.training = training
        self.seed = seed
        self.patch_size = patch_size
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.sigma_json_path = sigma_json_path
        self.load_sigma_json_if_exists = load_sigma_json_if_exists

        with h5py.File(self.h5_path, "r") as h5_file:
            if self.group_name not in h5_file:
                raise KeyError(f"Group '{self.group_name}' not found in {self.h5_path}")

            group = h5_file[self.group_name]

            if dataset_paths is None:
                self.dataset_paths = [f"{self.group_name}/{name}" for name in group.keys()]
            else:
                self.dataset_paths = dataset_paths

            if len(self.dataset_paths) == 0:
                raise ValueError("No datasets found in the selected H5 group.")

            first_cube = np.array(h5_file[self.dataset_paths[0]], dtype=np.float32)
            if first_cube.ndim != 3:
                raise ValueError(
                    f"Expected hyperspectral cube with shape (H, W, B), "
                    f"got {first_cube.shape} from {self.dataset_paths[0]}"
                )

            _, _, num_bands = first_cube.shape

        if band_indices is None:
            self.band_indices = list(range(num_bands))
        else:
            self.band_indices = list(band_indices)

        if len(self.band_indices) == 0:
            raise ValueError("band_indices is empty.")

        # One sample = one (cube, band) pair
        self.samples = []
        for dataset_path in self.dataset_paths:
            for band_idx in self.band_indices:
                self.samples.append((dataset_path, band_idx))

        # Generate or load ONE sigma value per band
        self.band_sigmas = self._build_or_load_band_sigmas()

    def _build_or_load_band_sigmas(self) -> dict[int, float]:
        """
        Build one fixed sigma per band.
        If a JSON file exists and loading is enabled, load from it.
        Otherwise generate and optionally save.
        """
        if (
            self.sigma_json_path is not None
            and self.load_sigma_json_if_exists
            and os.path.isfile(self.sigma_json_path)
        ):
            with open(self.sigma_json_path, "r") as f:
                data = json.load(f)

            if isinstance(data, dict):
                band_sigmas = {int(k): float(v) for k, v in data.items()}
            elif isinstance(data, list):
                band_sigmas = {i: float(v) for i, v in enumerate(data)}
            else:
                raise ValueError("Sigma JSON must be a dict or list.")

            # Check required bands exist
            for b in self.band_indices:
                if b not in band_sigmas:
                    raise KeyError(f"Band {b} is missing from sigma JSON.")

            return band_sigmas

        # Otherwise generate one random-but-fixed sigma per band
        rng = np.random.default_rng(self.seed)
        band_sigmas = {
            int(b): float(rng.uniform(self.sigma_min, self.sigma_max))
            for b in self.band_indices
        }

        if self.sigma_json_path is not None:
            with open(self.sigma_json_path, "w") as f:
                json.dump({str(k): v for k, v in band_sigmas.items()}, f, indent=2)

        return band_sigmas

    def __len__(self) -> int:
        return len(self.samples)

    def _load_band(self, dataset_path: str, band_idx: int) -> np.ndarray:
        with h5py.File(self.h5_path, "r") as h5_file:
            cube = np.array(h5_file[dataset_path], dtype=np.float32)

        if cube.ndim != 3:
            raise ValueError(f"Expected cube shape (H, W, B), got {cube.shape} for {dataset_path}")

        band = cube[:, :, band_idx]

        if self.normalize:
            band_min = band.min()
            band_max = band.max()
            if band_max > band_min:
                band = (band - band_min) / (band_max - band_min)
            else:
                band = np.zeros_like(band, dtype=np.float32)

        return band.astype(np.float32)

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

    def _generate_noise(self, shape, sigma_value: float, index: int) -> np.ndarray:
        if self.training:
            noise = np.random.normal(0.0, sigma_value, size=shape).astype(np.float32)
        else:
            rng = np.random.default_rng(self.seed + index)
            noise = rng.normal(0.0, sigma_value, size=shape).astype(np.float32)
        return noise

    def __getitem__(self, index: int):
        dataset_path, band_idx = self.samples[index]

        clean = self._load_band(dataset_path, band_idx)
        clean = self._extract_patch(clean, index)

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
            "sigma_map": torch.from_numpy(sigma_map.astype(np.float32)),
            "noise": torch.from_numpy(noise.astype(np.float32)),
            "clean": torch.from_numpy(clean.astype(np.float32)),
            "sigma": torch.tensor([sigma_value], dtype=torch.float32),
            "sigma_raw": torch.tensor([sigma_raw], dtype=torch.float32),
            "band_idx": band_idx,
            "key": f"{dataset_path}::band_{band_idx}",
            "dataset_path": dataset_path,
        }