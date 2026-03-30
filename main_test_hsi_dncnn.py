import os
import glob
import json
import logging
import argparse
from collections import OrderedDict

import numpy as np
import torch

from utils import utils_logger
from utils import utils_model
from utils import utils_option as option


try:
    from skimage.metrics import structural_similarity as skimage_ssim
except Exception:
    skimage_ssim = None


"""
HSI test script for HSIDnCNN.

What this fixes compared with the original KAIR/DnCNN test script:
- loads HSIDnCNN instead of grayscale/color DnCNN
- reads hyperspectral cubes instead of 2D image files
- denoises a full cube band-by-band using spectral neighbors
- supports .npy/.npz/.mat/.h5/.hdf5/.tif/.tiff
- uses the HSI options from options/train_hsi_dncnn.json
- can evaluate full-cube PSNR/SSIM band-by-band and save denoised cubes
"""


def str2bool(v):
    if isinstance(v, bool):
        return v
    v = str(v).strip().lower()
    if v in {'true', '1', 'yes', 'y', 'on'}:
        return True
    if v in {'false', '0', 'no', 'n', 'off'}:
        return False
    raise argparse.ArgumentTypeError(f'Invalid boolean value: {v}')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, default='options/train_hsi_dncnn.json',
                        help='Path to the HSI option JSON file.')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Explicit checkpoint path. If omitted, the latest *_G.pth under the task/models folder is used.')
    parser.add_argument('--results', type=str, default=None,
                        help='Output directory. Default: <task>/test_hsi_dncnn')
    parser.add_argument('--need_degradation', type=str2bool, default=True,
                        help='If true and no noisy cubes are provided, create synthetic noisy cubes from clean cubes.')
    parser.add_argument('--x8', type=str2bool, default=False,
                        help='Use x8 self-ensemble during inference.')
    parser.add_argument('--save_noisy', type=str2bool, default=False,
                        help='Also save noisy cubes as .npy.')
    parser.add_argument('--save_clean', type=str2bool, default=False,
                        help='Also save clean cubes as .npy when available.')
    parser.add_argument('--suffix', type=str, default='',
                        help='Optional suffix appended to the result folder name.')
    return parser.parse_args()


# -----------------------------------------------------------------------------
# File IO helpers
# -----------------------------------------------------------------------------
def get_cube_paths(dataroot, cube_ext=None):
    if dataroot is None:
        return []
    if os.path.isfile(dataroot):
        return [dataroot]
    if not os.path.isdir(dataroot):
        raise ValueError(f'Path does not exist: {dataroot}')

    if cube_ext is None or cube_ext == '*':
        patterns = ['*.npy', '*.npz', '*.mat', '*.h5', '*.hdf5', '*.tif', '*.tiff']
    else:
        cube_ext = cube_ext.lower()
        patterns = [f'*{cube_ext}'] if cube_ext.startswith('.') else [f'*.{cube_ext}']

    paths = []
    for p in patterns:
        paths.extend(glob.glob(os.path.join(dataroot, p)))
    return sorted(paths)


def match_noisy_paths(clean_paths, noisy_paths):
    if not noisy_paths:
        return {os.path.splitext(os.path.basename(p))[0]: None for p in clean_paths}

    noisy_map = {os.path.splitext(os.path.basename(p))[0]: p for p in noisy_paths}
    matched = {}
    for p in clean_paths:
        stem = os.path.splitext(os.path.basename(p))[0]
        matched[stem] = noisy_map.get(stem, None)
    return matched


# -----------------------------------------------------------------------------
# Hyperspectral cube loading / formatting
# -----------------------------------------------------------------------------
def load_cube(path, cube_key=None):
    ext = os.path.splitext(path)[1].lower()

    if ext == '.npy':
        cube = np.load(path)

    elif ext == '.npz':
        npz_file = np.load(path)
        if cube_key is not None:
            cube = npz_file[cube_key]
        else:
            first_key = list(npz_file.keys())[0]
            cube = npz_file[first_key]

    elif ext == '.mat':
        from scipy.io import loadmat
        mat = loadmat(path)
        cube = pick_3d_array_from_dict(mat, cube_key, path)

    elif ext in ['.h5', '.hdf5']:
        import h5py
        with h5py.File(path, 'r') as f:
            if cube_key is not None:
                cube = np.array(f[cube_key])
            else:
                cube = pick_3d_array_from_h5(f, path)

    elif ext in ['.tif', '.tiff']:
        import tifffile
        cube = tifffile.imread(path)

    else:
        raise ValueError(f'Unsupported hyperspectral file format: {ext}')

    cube = np.asarray(cube)
    if cube.ndim != 3:
        raise ValueError(f'Cube at {path} must be 3D, but got shape {cube.shape}')
    return cube


def pick_3d_array_from_dict(data_dict, cube_key, path):
    if cube_key is not None:
        if cube_key not in data_dict:
            raise KeyError(f'cube_key="{cube_key}" not found in {path}')
        arr = np.asarray(data_dict[cube_key])
        if arr.ndim != 3:
            raise ValueError(f'Variable "{cube_key}" in {path} is not 3D.')
        return arr

    candidates = []
    for k, v in data_dict.items():
        arr = np.asarray(v)
        if arr.ndim == 3 and not k.startswith('__'):
            candidates.append(arr)

    if not candidates:
        raise ValueError(f'No 3D array found in {path}')

    candidates.sort(key=lambda x: np.prod(x.shape), reverse=True)
    return candidates[0]


def pick_3d_array_from_h5(h5file, path):
    candidates = []

    def visitor(_name, obj):
        if hasattr(obj, 'shape') and len(obj.shape) == 3:
            candidates.append(np.array(obj))

    h5file.visititems(visitor)
    if not candidates:
        raise ValueError(f'No 3D dataset found in {path}')

    candidates.sort(key=lambda x: np.prod(x.shape), reverse=True)
    return candidates[0]


def to_hwb(cube, band_dim='auto'):
    cube = np.asarray(cube)
    if cube.ndim != 3:
        raise ValueError(f'Expected 3D cube, got shape {cube.shape}')

    if band_dim in [0, 'first']:
        cube = np.moveaxis(cube, 0, -1)
    elif band_dim in [1, 'middle']:
        cube = np.moveaxis(cube, 1, -1)
    elif band_dim in [2, 'last']:
        pass
    elif band_dim == 'auto':
        bdim = int(np.argmin(list(cube.shape)))
        cube = np.moveaxis(cube, bdim, -1)
    else:
        raise ValueError(f'Unsupported band_dim: {band_dim}')

    return cube


def normalize_cube(cube):
    """
    Map cube to float32 in [0, 1].
    Fixes the uint16 logic bug by checking the original dtype before astype(float32).
    """
    orig_dtype = cube.dtype
    cube = cube.astype(np.float32)

    cmin = float(cube.min())
    cmax = float(cube.max())

    if cmax <= 1.0 and cmin >= 0.0:
        return cube

    if np.issubdtype(orig_dtype, np.integer):
        if cmin >= 0.0 and cmax <= 255.0:
            return cube / 255.0
        if cmin >= 0.0 and cmax <= 65535.0:
            return cube / 65535.0

    if cmin >= 0.0 and cmax <= 255.0:
        return cube / 255.0

    denom = max(cmax - cmin, 1e-12)
    return (cube - cmin) / denom


# -----------------------------------------------------------------------------
# Spectral-window helpers
# -----------------------------------------------------------------------------
def fix_index(idx, num_bands, boundary_mode):
    if boundary_mode == 'reflect':
        while idx < 0 or idx >= num_bands:
            if idx < 0:
                idx = -idx
            if idx >= num_bands:
                idx = 2 * num_bands - 2 - idx
        return idx

    if boundary_mode == 'replicate':
        return min(max(idx, 0), num_bands - 1)

    if boundary_mode == 'wrap':
        return idx % num_bands

    raise ValueError(f'Unsupported boundary_mode: {boundary_mode}')


def get_neighbor_indices(target_band, num_bands, in_bands, boundary_mode):
    half = in_bands // 2
    raw_indices = list(range(target_band - half, target_band + half + 1))

    if boundary_mode in ['valid', 'skip']:
        if min(raw_indices) < 0 or max(raw_indices) >= num_bands:
            return None
        return raw_indices

    return [fix_index(i, num_bands, boundary_mode) for i in raw_indices]


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------
def calculate_psnr_float(img1, img2, border=0):
    if border > 0:
        img1 = img1[border:-border, border:-border]
        img2 = img2[border:-border, border:-border]
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse <= 0:
        return float('inf')
    return 10.0 * np.log10(1.0 / mse)


def calculate_ssim_float(img1, img2, border=0):
    if skimage_ssim is None:
        return None
    if border > 0:
        img1 = img1[border:-border, border:-border]
        img2 = img2[border:-border, border:-border]
    return float(skimage_ssim(img1.astype(np.float64), img2.astype(np.float64), data_range=1.0))


# -----------------------------------------------------------------------------
# Model helpers
# -----------------------------------------------------------------------------
def build_model(opt, center_idx, device):
    net_opt = opt['netG']
    if net_opt['net_type'].lower() != 'hsi_dncnn':
        raise ValueError(f"This script expects net_type='hsi_dncnn', got {net_opt['net_type']}")

    from models.network_dncnn import HSIDnCNN as net
    model = net(
        in_nc=net_opt['in_nc'],
        out_nc=net_opt['out_nc'],
        nc=net_opt['nc'],
        nb=net_opt['nb'],
        act_mode=net_opt['act_mode'],
        center_idx=center_idx,
    )
    model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def load_checkpoint(model, model_path, device):
    checkpoint = torch.load(model_path, map_location=device)

    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # handle DataParallel checkpoints
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=True)
    return model


def resolve_model_path(opt, override_model_path):
    if override_model_path is not None:
        return override_model_path

    pretrained = opt['path'].get('pretrained_netG', None)
    models_dir = os.path.join(opt['path']['task'], 'models')
    _, model_path = option.find_last_checkpoint(models_dir, net_type='G', pretrained_path=pretrained)

    if model_path is None or not os.path.isfile(model_path):
        raise FileNotFoundError(
            'Could not find a generator checkpoint. Pass --model_path explicitly or make sure training produced *_G.pth files.'
        )
    return model_path


# -----------------------------------------------------------------------------
# Full-cube denoising
# -----------------------------------------------------------------------------
def prepare_test_settings(opt):
    test_opt = opt['datasets']['test']

    cube_ext = test_opt.get('cube_ext', None)
    data_format = test_opt.get('data_format', None)
    if cube_ext is None:
        if data_format is None:
            cube_ext = '.npy'
        else:
            data_format = str(data_format).strip().lower()
            cube_ext = data_format if data_format.startswith('.') else f'.{data_format}'

    if 'in_bands' in test_opt:
        in_bands = int(test_opt['in_bands'])
    elif 'input_bands' in test_opt:
        in_bands = int(test_opt['input_bands'])
    elif 'neighbor_radius' in test_opt:
        in_bands = 2 * int(test_opt['neighbor_radius']) + 1
    else:
        in_bands = int(opt['netG']['in_nc'])

    if 'center_idx' in test_opt:
        center_idx = int(test_opt['center_idx'])
    else:
        target_position = test_opt.get('target_position', 'center')
        if isinstance(target_position, str):
            tp = target_position.strip().lower()
            if tp == 'center':
                center_idx = in_bands // 2
            elif tp == 'left':
                center_idx = 0
            elif tp == 'right':
                center_idx = in_bands - 1
            else:
                center_idx = int(tp)
        else:
            center_idx = int(target_position)

    sigma_unit = str(test_opt.get('sigma_unit', '255')).strip().lower()
    if sigma_unit in ['255', 'image', 'uint8']:
        sigma_scale = 255.0
    elif sigma_unit in ['1', 'normalized', 'float']:
        sigma_scale = 1.0
    else:
        raise ValueError(f'Unsupported sigma_unit: {sigma_unit}')

    if 'same_sigma_within_sample' in test_opt:
        same_sigma = bool(test_opt['same_sigma_within_sample'])
    else:
        same_sigma = bool(test_opt.get('same_sigma_for_all_5_bands', False))

    settings = {
        'clean_root': test_opt.get('dataroot_H', None),
        'noisy_root': test_opt.get('dataroot_L', None),
        'cube_ext': cube_ext,
        'cube_key': test_opt.get('cube_key', None),
        'band_dim': test_opt.get('band_dim', 'auto'),
        'in_bands': in_bands,
        'center_idx': center_idx,
        'boundary_mode': str(test_opt.get('boundary_mode', 'reflect')).lower(),
        'sigma_min': float(test_opt.get('sigma_min', 0)),
        'sigma_max': float(test_opt.get('sigma_max', 50)),
        'sigma_scale': sigma_scale,
        'same_sigma': same_sigma,
        'seed': int(test_opt.get('seed', 0) or 0),
    }
    return settings


def make_noisy_cube(clean_cube, sigma_per_band, sigma_scale, rng):
    noisy_cube = clean_cube.copy().astype(np.float32)
    for b in range(noisy_cube.shape[-1]):
        noise = rng.normal(0.0, sigma_per_band[b] / sigma_scale, size=noisy_cube[:, :, b].shape).astype(np.float32)
        noisy_cube[:, :, b] += noise
    return np.clip(noisy_cube, 0.0, 1.0)


@torch.no_grad()
def denoise_full_cube(model, noisy_cube, in_bands, boundary_mode, x8, device):
    H, W, B = noisy_cube.shape
    denoised = np.zeros_like(noisy_cube, dtype=np.float32)
    processed_mask = np.zeros(B, dtype=bool)

    for b in range(B):
        band_indices = get_neighbor_indices(b, B, in_bands, boundary_mode)

        # valid/skip mode: keep outer bands unchanged because there is no full neighborhood
        if band_indices is None:
            denoised[:, :, b] = noisy_cube[:, :, b]
            continue

        stack = noisy_cube[:, :, band_indices].astype(np.float32)  # [H, W, C]
        stack = np.transpose(stack, (2, 0, 1))                     # [C, H, W]
        stack = torch.from_numpy(stack).unsqueeze(0).to(device)    # [1, C, H, W]

        if not x8:
            pred = model(stack)
        else:
            pred = utils_model.test_mode(model, stack, mode=3)

        denoised[:, :, b] = pred.squeeze().detach().float().cpu().numpy()
        processed_mask[b] = True

    denoised = np.clip(denoised, 0.0, 1.0)
    return denoised, processed_mask


def save_cube_npy(path, cube):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, cube)


def main():
    args = parse_args()

    # Use is_train=True so KAIR builds the task/models paths exactly the same way as training.
    opt = option.parse(args.opt, is_train=True)
    test_settings = prepare_test_settings(opt)

    result_name = f"test_hsi_dncnn{args.suffix}" if args.suffix else 'test_hsi_dncnn'
    result_dir = args.results or os.path.join(opt['path']['task'], result_name)
    os.makedirs(result_dir, exist_ok=True)

    logger_name = 'test_hsi_dncnn'
    utils_logger.logger_info(logger_name, os.path.join(result_dir, f'{logger_name}.log'))
    logger = logging.getLogger(logger_name)
    logger.info(option.dict2str(opt))
    logger.info('Test settings: %s', json.dumps(test_settings, indent=2))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = resolve_model_path(opt, args.model_path)

    model = build_model(opt, test_settings['center_idx'], device)
    model = load_checkpoint(model, model_path, device)
    logger.info('Model path: %s', model_path)
    logger.info('Params number: %d', sum(p.numel() for p in model.parameters()))

    clean_paths = get_cube_paths(test_settings['clean_root'], test_settings['cube_ext']) if test_settings['clean_root'] else []
    noisy_paths = get_cube_paths(test_settings['noisy_root'], test_settings['cube_ext']) if test_settings['noisy_root'] else []

    if not clean_paths and not noisy_paths:
        raise ValueError('No test cubes found. Check datasets.test.dataroot_H / dataroot_L in your JSON.')

    if clean_paths:
        pair_map = match_noisy_paths(clean_paths, noisy_paths)
        stems = [os.path.splitext(os.path.basename(p))[0] for p in clean_paths]
    else:
        # inference-only case: noisy cubes without ground truth
        pair_map = {os.path.splitext(os.path.basename(p))[0]: p for p in noisy_paths}
        stems = list(pair_map.keys())

    global_psnr = []
    global_ssim = []
    metrics_rows = []

    for idx, stem in enumerate(stems):
        clean_path = None
        noisy_path = None

        if clean_paths:
            clean_path = clean_paths[idx]
            noisy_path = pair_map[stem]
        else:
            noisy_path = pair_map[stem]

        logger.info('Processing cube %d/%d: %s', idx + 1, len(stems), stem)

        clean_cube = None
        if clean_path is not None:
            clean_cube = normalize_cube(to_hwb(load_cube(clean_path, test_settings['cube_key']), test_settings['band_dim']))

        if noisy_path is not None:
            noisy_cube = normalize_cube(to_hwb(load_cube(noisy_path, test_settings['cube_key']), test_settings['band_dim']))
            sigma_per_band = None
        else:
            if clean_cube is None:
                raise ValueError(f'No clean or noisy cube available for {stem}.')

            rng = np.random.default_rng(test_settings['seed'] + idx)
            if args.need_degradation:
                if test_settings['same_sigma']:
                    sigma = float(rng.uniform(test_settings['sigma_min'], test_settings['sigma_max']))
                    sigma_per_band = np.full(clean_cube.shape[-1], sigma, dtype=np.float32)
                else:
                    sigma_per_band = rng.uniform(
                        test_settings['sigma_min'],
                        test_settings['sigma_max'],
                        size=clean_cube.shape[-1]
                    ).astype(np.float32)
                noisy_cube = make_noisy_cube(clean_cube, sigma_per_band, test_settings['sigma_scale'], rng)
            else:
                sigma_per_band = np.zeros(clean_cube.shape[-1], dtype=np.float32)
                noisy_cube = clean_cube.copy()

        if clean_cube is not None and noisy_cube.shape != clean_cube.shape:
            raise ValueError(
                f'Shape mismatch for {stem}: noisy {noisy_cube.shape} vs clean {clean_cube.shape}'
            )

        denoised_cube, processed_mask = denoise_full_cube(
            model=model,
            noisy_cube=noisy_cube,
            in_bands=test_settings['in_bands'],
            boundary_mode=test_settings['boundary_mode'],
            x8=args.x8,
            device=device,
        )

        save_cube_npy(os.path.join(result_dir, f'{stem}_denoised.npy'), denoised_cube)
        if args.save_noisy:
            save_cube_npy(os.path.join(result_dir, f'{stem}_noisy.npy'), noisy_cube)
        if args.save_clean and clean_cube is not None:
            save_cube_npy(os.path.join(result_dir, f'{stem}_clean.npy'), clean_cube)

        row = OrderedDict()
        row['name'] = stem
        row['bands'] = int(noisy_cube.shape[-1])
        row['processed_bands'] = int(processed_mask.sum())
        row['model_path'] = model_path
        row['boundary_mode'] = test_settings['boundary_mode']

        if sigma_per_band is not None:
            row['sigma_mean'] = float(np.mean(sigma_per_band))
            row['sigma_min'] = float(np.min(sigma_per_band))
            row['sigma_max'] = float(np.max(sigma_per_band))

        if clean_cube is not None:
            band_psnr = []
            band_ssim = []
            valid_band_ids = np.where(processed_mask)[0].tolist()
            if len(valid_band_ids) == 0:
                valid_band_ids = list(range(clean_cube.shape[-1]))

            for b in valid_band_ids:
                psnr = calculate_psnr_float(denoised_cube[:, :, b], clean_cube[:, :, b])
                band_psnr.append(psnr)
                ssim = calculate_ssim_float(denoised_cube[:, :, b], clean_cube[:, :, b])
                if ssim is not None:
                    band_ssim.append(ssim)

            row['avg_psnr'] = float(np.mean(band_psnr)) if band_psnr else None
            row['avg_ssim'] = float(np.mean(band_ssim)) if band_ssim else None
            row['valid_band_range'] = [int(valid_band_ids[0]), int(valid_band_ids[-1])] if valid_band_ids else None

            global_psnr.extend(band_psnr)
            global_ssim.extend(band_ssim)

            logger.info('%s - avg PSNR: %.4f dB%s',
                        stem,
                        row['avg_psnr'],
                        '' if row['avg_ssim'] is None else f'; avg SSIM: {row["avg_ssim"]:.6f}')
        else:
            row['avg_psnr'] = None
            row['avg_ssim'] = None
            row['valid_band_range'] = None
            logger.info('%s - inference only (no ground truth provided).', stem)

        metrics_rows.append(row)

    summary = OrderedDict()
    summary['model_path'] = model_path
    summary['results_dir'] = result_dir
    summary['num_cubes'] = len(metrics_rows)
    summary['global_avg_psnr'] = float(np.mean(global_psnr)) if global_psnr else None
    summary['global_avg_ssim'] = float(np.mean(global_ssim)) if global_ssim else None
    summary['per_cube'] = metrics_rows

    with open(os.path.join(result_dir, 'metrics_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info('Results saved to: %s', result_dir)
    if summary['global_avg_psnr'] is not None:
        logger.info('Global average PSNR: %.4f dB', summary['global_avg_psnr'])
    if summary['global_avg_ssim'] is not None:
        logger.info('Global average SSIM: %.6f', summary['global_avg_ssim'])


if __name__ == '__main__':
    main()