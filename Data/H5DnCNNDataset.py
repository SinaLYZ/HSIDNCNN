import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class H5DnCNNDataset(Dataset):
    """
    DnCNN dataset for hyperspectral H5 data, using each spectral band
    as an independent grayscale image.

    H5 structure expected:
        root/
            hsi_27/
                sample1.npy   -> shape (H, W, B)
                sample2.npy   -> shape (H, W, B)
                ...

    Example:
        sample shape = (1080, 2048, 151)

    This dataset:
    - opens all datasets inside a given H5 group
    - treats each spectral band as one grayscale image
    - optionally extracts random patches during training
    - adds Gaussian noise on-the-fly
    - returns noisy / noise / clean tensors shaped [1, H, W]

    Args:
        h5_path (str): Path to H5 file.
        group_name (str): Name of group inside H5, e.g. "hsi_27".
        sigma (float): Noise std in pixel scale [0,255].
        normalize (bool): Whether to normalize image values.
        training (bool): If True, use random noise and random patches.
        seed (int): Seed for deterministic test-time noise/patching.
        patch_size (int or None): Patch size for band crops. If None, use full band.
        band_indices (list[int] or None): Optional subset of spectral bands.
        dataset_paths (list[str] or None): Optional subset of dataset paths inside the group.
    """

    def __init__(
        self,
        h5_path: str,
        group_name: str = "hsi_27",
        sigma: float = 25.0,
        normalize: bool = True,
        training: bool = True,
        seed: int = 42,
        patch_size: int | None = 64,
        band_indices=None,
        dataset_paths=None,
    ) -> None:
        super().__init__()

        self.h5_path = h5_path
        self.group_name = group_name
        self.normalize = normalize
        self.training = training
        self.seed = seed
        self.patch_size = patch_size

        self.sigma = sigma / 255.0 if normalize else sigma

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

            # Inspect first cube to determine number of bands
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

        # Each item = one (cube, band) pair
        self.samples = []
        for dataset_path in self.dataset_paths:
            for band_idx in self.band_indices:
                self.samples.append((dataset_path, band_idx))

    def __len__(self) -> int:
        return len(self.samples)

    def _load_band(self, dataset_path: str, band_idx: int) -> np.ndarray:
        with h5py.File(self.h5_path, "r") as h5_file:
            cube = np.array(h5_file[dataset_path], dtype=np.float32)

        # cube shape: (H, W, B)
        if cube.ndim != 3:
            raise ValueError(f"Expected cube shape (H, W, B), got {cube.shape} for {dataset_path}")

        band = cube[:, :, band_idx]  # shape: (H, W)

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
            raise ValueError(
                f"patch_size={ps} is larger than band shape {(h, w)}"
            )

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
        dataset_path, band_idx = self.samples[index]

        clean = self._load_band(dataset_path, band_idx)      # (H, W)
        clean = self._extract_patch(clean, index)            # (ps, ps) or full image

        noise = self._generate_noise(clean.shape, index)     # (H, W)
        noisy = clean + noise

        if self.normalize:
            noisy = np.clip(noisy, 0.0, 1.0)

        # Convert to [1, H, W] for grayscale DnCNN
        clean = np.expand_dims(clean, axis=0)
        noise = np.expand_dims(noise, axis=0)
        noisy = np.expand_dims(noisy, axis=0)

        return {
            "noisy": torch.from_numpy(noisy.astype(np.float32)),
            "noise": torch.from_numpy(noise.astype(np.float32)),
            "clean": torch.from_numpy(clean.astype(np.float32)),
            "key": f"{dataset_path}::band_{band_idx}",
            "dataset_path": dataset_path,
            "band_idx": band_idx,
        }