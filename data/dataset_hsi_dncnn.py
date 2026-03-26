import os
import glob
import random
import numpy as np
import torch
import torch.utils.data as data
import utils.utils_image as util


class DatasetHSIDnCNN(data.Dataset):
    """
    HSI dataset for KAIR-style DnCNN training.

    Input:
        noisy stack of neighboring bands, shape [in_bands, H, W]
        example for in_bands=5: [b-2, b-1, b, b+1, b+2]

    Target:
        clean target band only, shape [1, H, W]

    Expected options in JSON:
        {
            "dataset_type": "hsi_dncnn",
            "dataroot_H": "trainsets/hsi_train",
            "cube_ext": ".npy",
            "H_size": 40,
            "in_bands": 5,
            "center_idx": 2,
            "boundary_mode": "reflect",
            "sigma_min": 5,
            "sigma_max": 50,
            "same_sigma_within_sample": false,
            "phase": "train" / "test"
        }

    Notes:
    - This file is written to match KAIR's DatasetDnCNN style.
    - It assumes hyperspectral cubes are stored one cube per file.
    - Best default format: .npy with shape [H, W, B] or [B, H, W].
    """

    def __init__(self, opt):
        super(DatasetHSIDnCNN, self).__init__()
        print('Dataset: HSI denoising with neighboring bands and band-dependent AWGN.')

        self.opt = opt
        self.phase = opt['phase']

        self.patch_size = opt.get('H_size', 64)
        self.in_bands = opt.get('in_bands', 5)
        self.center_idx = opt.get('center_idx', self.in_bands // 2)
        self.boundary_mode = opt.get('boundary_mode', 'reflect').lower()

        self.sigma_min = opt.get('sigma_min', 5)
        self.sigma_max = opt.get('sigma_max', 50)
        self.same_sigma_within_sample = opt.get('same_sigma_within_sample', False)

        self.cube_ext = opt.get('cube_ext', '.npy')
        self.cube_key = opt.get('cube_key', None)
        self.band_dim = opt.get('band_dim', 'auto')   # auto / first / last / middle / 0 / 1 / 2
        self.seed = opt.get('seed', 0)

        assert self.in_bands % 2 == 1, 'in_bands must be odd.'
        assert 0 <= self.center_idx < self.in_bands, 'center_idx must be within input band range.'
        assert self.sigma_min >= 0 and self.sigma_max >= self.sigma_min, 'Invalid sigma range.'
        assert self.boundary_mode in ['reflect', 'replicate', 'wrap', 'valid', 'skip'], \
            "boundary_mode must be one of: reflect, replicate, wrap, valid, skip"

        self.paths_H = self._get_cube_paths(opt['dataroot_H'], self.cube_ext)
        if len(self.paths_H) == 0:
            raise ValueError(f'No hyperspectral cube files found in: {opt["dataroot_H"]}')

    def __getitem__(self, index):
        H_path = self.paths_H[index]
        L_path = H_path

        cube_H = self._load_cube(H_path)
        cube_H = self._to_hwb(cube_H)
        cube_H = self._normalize_cube(cube_H)

        H, W, B = cube_H.shape
        if B < self.in_bands:
            raise ValueError(
                f'Cube at {H_path} has only {B} bands, but in_bands={self.in_bands}.'
            )

        # -------------------------------
        # training: random crop
        # -------------------------------
        if self.phase == 'train':
            rnd_h = random.randint(0, max(0, H - self.patch_size))
            rnd_w = random.randint(0, max(0, W - self.patch_size))
            cube_patch = cube_H[
                rnd_h:rnd_h + self.patch_size,
                rnd_w:rnd_w + self.patch_size,
                :
            ]

            # augmentation (same style as KAIR)
            mode = random.randint(0, 7)
            cube_patch = util.augment_img(cube_patch, mode=mode)

            rng = np.random.default_rng()
        else:
            cube_patch = cube_H
            rng = np.random.default_rng(self.seed + index)

        _, _, Bp = cube_patch.shape

        # -------------------------------
        # choose target band
        # -------------------------------
        target_band = self._sample_target_band(Bp, rng)
        band_indices = self._get_neighbor_indices(target_band, Bp)

        stack_H = cube_patch[:, :, band_indices].astype(np.float32)             # [H, W, in_bands]
        target_H = cube_patch[:, :, target_band:target_band + 1].astype(np.float32)  # [H, W, 1]

        # -------------------------------
        # band-dependent Gaussian noise
        # -------------------------------
        sigmas = self._sample_sigmas(rng)  # in 8-bit scale, e.g. [5, 50]
        stack_L = stack_H.copy()

        for i in range(self.in_bands):
            noise = rng.normal(
                loc=0.0,
                scale=sigmas[i] / 255.0,
                size=stack_L[:, :, i].shape
            ).astype(np.float32)
            stack_L[:, :, i] += noise

        # -------------------------------
        # numpy(HWC) -> tensor(CHW)
        # -------------------------------
        img_L = util.single2tensor3(stack_L)
        img_H = util.single2tensor3(target_H)

        return {
            'L': img_L,
            'H': img_H,
            'H_path': H_path,
            'L_path': L_path,
            'target_band': target_band,
            'band_indices': np.array(band_indices, dtype=np.int64),
            'sigmas': np.array(sigmas, dtype=np.float32)
        }

    def __len__(self):
        return len(self.paths_H)

    # -------------------------------------------------------------------------
    # helpers
    # -------------------------------------------------------------------------
    def _get_cube_paths(self, dataroot, cube_ext):
        if dataroot is None:
            return []

        if os.path.isfile(dataroot):
            return [dataroot]

        if not os.path.isdir(dataroot):
            raise ValueError(f'dataroot_H does not exist: {dataroot}')

        if cube_ext is None or cube_ext == '*':
            patterns = ['*.npy', '*.npz', '*.mat', '*.h5', '*.hdf5', '*.tif', '*.tiff']
        else:
            cube_ext = cube_ext.lower()
            if cube_ext.startswith('.'):
                patterns = [f'*{cube_ext}']
            else:
                patterns = [f'*.{cube_ext}']

        paths = []
        for p in patterns:
            paths.extend(glob.glob(os.path.join(dataroot, p)))

        return sorted(paths)

    def _load_cube(self, path):
        ext = os.path.splitext(path)[1].lower()

        if ext == '.npy':
            cube = np.load(path)

        elif ext == '.npz':
            npz_file = np.load(path)
            if self.cube_key is not None:
                cube = npz_file[self.cube_key]
            else:
                first_key = list(npz_file.keys())[0]
                cube = npz_file[first_key]

        elif ext == '.mat':
            try:
                from scipy.io import loadmat
            except ImportError as e:
                raise ImportError('scipy is required to read .mat files.') from e

            mat = loadmat(path)
            cube = self._pick_3d_array_from_dict(mat, self.cube_key, path)

        elif ext in ['.h5', '.hdf5']:
            try:
                import h5py
            except ImportError as e:
                raise ImportError('h5py is required to read .h5/.hdf5 files.') from e

            with h5py.File(path, 'r') as f:
                if self.cube_key is not None:
                    cube = np.array(f[self.cube_key])
                else:
                    cube = self._pick_3d_array_from_h5(f, path)

        elif ext in ['.tif', '.tiff']:
            try:
                import tifffile
            except ImportError as e:
                raise ImportError('tifffile is required to read .tif/.tiff files.') from e

            cube = tifffile.imread(path)

        else:
            raise ValueError(f'Unsupported hyperspectral file format: {ext}')

        cube = np.asarray(cube)
        if cube.ndim != 3:
            raise ValueError(f'Cube at {path} must be 3D, but got shape {cube.shape}')

        return cube

    def _pick_3d_array_from_dict(self, data_dict, cube_key, path):
        if cube_key is not None:
            if cube_key not in data_dict:
                raise KeyError(f'cube_key="{cube_key}" not found in {path}')
            cube = data_dict[cube_key]
            if np.asarray(cube).ndim != 3:
                raise ValueError(f'Variable "{cube_key}" in {path} is not 3D.')
            return cube

        candidates = []
        for k, v in data_dict.items():
            arr = np.asarray(v)
            if arr.ndim == 3 and not k.startswith('__'):
                candidates.append((k, arr))

        if len(candidates) == 0:
            raise ValueError(f'No 3D array found in {path}')

        # choose largest 3D array
        candidates.sort(key=lambda x: np.prod(x[1].shape), reverse=True)
        return candidates[0][1]

    def _pick_3d_array_from_h5(self, h5file, path):
        candidates = []

        def visitor(name, obj):
            if hasattr(obj, 'shape') and len(obj.shape) == 3:
                candidates.append(np.array(obj))

        h5file.visititems(visitor)

        if len(candidates) == 0:
            raise ValueError(f'No 3D dataset found in {path}')

        candidates.sort(key=lambda x: np.prod(x.shape), reverse=True)
        return candidates[0]

    def _to_hwb(self, cube):
        """
        Convert cube to [H, W, B].
        """
        cube = np.asarray(cube)

        if cube.ndim != 3:
            raise ValueError(f'Expected 3D cube, got shape {cube.shape}')

        band_dim = self.band_dim

        if band_dim in [0, 'first']:
            cube = np.moveaxis(cube, 0, -1)
        elif band_dim in [1, 'middle']:
            cube = np.moveaxis(cube, 1, -1)
        elif band_dim in [2, 'last']:
            pass
        elif band_dim == 'auto':
            # assume the smallest dimension is spectral dimension
            dims = list(cube.shape)
            bdim = int(np.argmin(dims))
            cube = np.moveaxis(cube, bdim, -1)
        else:
            raise ValueError(f'Unsupported band_dim: {band_dim}')

        return cube.astype(np.float32)

    def _normalize_cube(self, cube):
        """
        Convert to float32 and roughly map into [0, 1].

        This is a generic fallback. If your dataset already uses a known scaling,
        replace this function with the exact normalization you need.
        """
        cube = cube.astype(np.float32)

        cmin = float(cube.min())
        cmax = float(cube.max())

        if cmax <= 1.0 and cmin >= 0.0:
            return cube

        if cmin >= 0.0 and cmax <= 255.0:
            return cube / 255.0

        if cmin >= 0.0 and cmax <= 65535.0 and np.issubdtype(cube.dtype, np.integer):
            return cube / 65535.0

        # generic fallback
        denom = max(cmax - cmin, 1e-12)
        return (cube - cmin) / denom

    def _sample_target_band(self, num_bands, rng):
        half = self.in_bands // 2

        if self.boundary_mode in ['valid', 'skip']:
            low = half
            high = num_bands - half - 1
            if low > high:
                raise ValueError(
                    f'Not enough bands ({num_bands}) for valid selection with in_bands={self.in_bands}'
                )
            return int(rng.integers(low, high + 1))

        return int(rng.integers(0, num_bands))

    def _get_neighbor_indices(self, target_band, num_bands):
        half = self.in_bands // 2
        raw_indices = list(range(target_band - half, target_band + half + 1))

        if self.boundary_mode in ['valid', 'skip']:
            if min(raw_indices) < 0 or max(raw_indices) >= num_bands:
                raise ValueError(
                    f'Invalid target_band={target_band} for valid mode with num_bands={num_bands}'
                )
            return raw_indices

        fixed = []
        for idx in raw_indices:
            fixed.append(self._fix_index(idx, num_bands))
        return fixed

    def _fix_index(self, idx, num_bands):
        if self.boundary_mode == 'reflect':
            # mirror around edges: -1->1, -2->2, B->B-2, ...
            while idx < 0 or idx >= num_bands:
                if idx < 0:
                    idx = -idx
                if idx >= num_bands:
                    idx = 2 * num_bands - 2 - idx
            return idx

        if self.boundary_mode == 'replicate':
            return min(max(idx, 0), num_bands - 1)

        if self.boundary_mode == 'wrap':
            return idx % num_bands

        raise ValueError(f'Unsupported boundary_mode: {self.boundary_mode}')

    def _sample_sigmas(self, rng):
        if self.same_sigma_within_sample:
            sigma = float(rng.uniform(self.sigma_min, self.sigma_max))
            return [sigma] * self.in_bands

        return [float(rng.uniform(self.sigma_min, self.sigma_max)) for _ in range(self.in_bands)]