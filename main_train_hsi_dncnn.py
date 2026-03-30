import os.path
import math
import argparse
import random
import logging
import traceback
from datetime import datetime
from pprint import pformat
from collections import OrderedDict

import numpy as np
import torch
from torch.utils.data import DataLoader

from utils import utils_logger
from utils import utils_image as util
from utils import utils_option as option

from data.select_dataset import define_Dataset
from models.select_model import define_Model


"""
HSI-friendly training entry point for DnCNN-style models.

Main fixes vs the original script:
1) Default option file points to the HSI config.
2) Validation is safer for HSI single-band outputs.
3) Metrics are computed on float arrays instead of quantized uint previews.
4) Saves .npy outputs during validation, with optional PNG previews.
5) Adds crash logging around the train / validation loop.
"""


def _debug_value(x):
    if torch.is_tensor(x):
        info = {
            'type': 'tensor',
            'shape': list(x.shape),
            'dtype': str(x.dtype),
            'device': str(x.device),
        }
        if x.numel() > 0:
            info['min'] = float(x.detach().min().item())
            info['max'] = float(x.detach().max().item())
        return info

    if isinstance(x, np.ndarray):
        info = {
            'type': 'ndarray',
            'shape': list(x.shape),
            'dtype': str(x.dtype),
        }
        if x.size > 0:
            info['min'] = float(np.min(x))
            info['max'] = float(np.max(x))
        return info

    if isinstance(x, (list, tuple)):
        return [_debug_value(v) for v in x]

    if isinstance(x, dict):
        return {k: _debug_value(v) for k, v in x.items()}

    return str(x)



def summarize_batch(data):
    if data is None:
        return None
    return {k: _debug_value(v) for k, v in data.items()}



def write_crash_dump(crash_log_path, stage, exc, **context):
    with open(crash_log_path, 'a', encoding='utf-8') as f:
        f.write('\n' + '=' * 100 + '\n')
        f.write(f'[{datetime.now().isoformat()}] {stage}\n')
        for k, v in context.items():
            f.write(f'{k}: {pformat(v)}\n')
        f.write('TRACEBACK:\n')
        f.write(''.join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
        f.write('\n')



def tensor_to_float_hw(x):
    """
    Convert tensor/array shaped like [1,H,W] or [H,W] to float32 [H,W].
    """
    if torch.is_tensor(x):
        x = x.detach().float().cpu().numpy()
    else:
        x = np.asarray(x, dtype=np.float32)

    x = np.squeeze(x)
    if x.ndim != 2:
        raise ValueError(f'Expected 2D image after squeeze, got shape {x.shape}')
    return x.astype(np.float32)



def calculate_psnr_float(img1, img2, data_range=1.0):
    img1 = np.asarray(img1, dtype=np.float32)
    img2 = np.asarray(img2, dtype=np.float32)
    mse = np.mean((img1 - img2) ** 2)
    if mse <= 1e-16:
        return float('inf')
    return float(20.0 * np.log10(data_range) - 10.0 * np.log10(mse))



def get_preview_uint(x):
    """
    Create a PNG-friendly uint preview from a float [H,W] image in [0,1].
    """
    x = np.clip(x, 0.0, 1.0)
    return np.uint8(np.round(x * 255.0))



def validate(model, test_loader, opt, current_step, epoch, logger, border=0):
    avg_psnr = 0.0
    avg_psnr_uint = 0.0
    idx = 0

    images_root = opt['path']['images']
    save_png_preview = bool(opt['datasets']['test'].get('save_png_preview', True))
    save_npy = bool(opt['datasets']['test'].get('save_npy', True))

    for test_data in test_loader:
        idx += 1

        image_name_ext = os.path.basename(test_data['L_path'][0])
        img_name, _ = os.path.splitext(image_name_ext)
        img_dir = os.path.join(images_root, img_name)
        util.mkdir(img_dir)

        model.feed_data(test_data)
        model.test()
        visuals = model.current_visuals()

        E_float = tensor_to_float_hw(visuals['E'])
        H_float = tensor_to_float_hw(visuals['H'])

        current_psnr = calculate_psnr_float(E_float, H_float, data_range=1.0)
        avg_psnr += current_psnr

        # Optional quantized preview metric for rough continuity with KAIR image logs.
        E_uint = get_preview_uint(E_float)
        H_uint = get_preview_uint(H_float)
        current_psnr_uint = util.calculate_psnr(E_uint, H_uint, border=border)
        avg_psnr_uint += current_psnr_uint

        if save_npy:
            np.save(os.path.join(img_dir, f'{img_name}_{current_step}_E.npy'), E_float)
            np.save(os.path.join(img_dir, f'{img_name}_{current_step}_H.npy'), H_float)

        if save_png_preview:
            util.imsave(E_uint, os.path.join(img_dir, f'{img_name}_{current_step}_E.png'))
            util.imsave(H_uint, os.path.join(img_dir, f'{img_name}_{current_step}_H.png'))

        extra_parts = []
        if 'target_band' in test_data:
            tb = test_data['target_band']
            if torch.is_tensor(tb):
                tb = int(tb.view(-1)[0].item())
            elif isinstance(tb, (list, tuple, np.ndarray)):
                tb = int(np.asarray(tb).reshape(-1)[0])
            else:
                tb = int(tb)
            extra_parts.append(f'target_band={tb}')

        if 'band_indices' in test_data:
            bi = test_data['band_indices']
            if torch.is_tensor(bi):
                bi = bi.detach().cpu().numpy()
            bi = np.asarray(bi).reshape(-1).tolist()
            extra_parts.append(f'band_indices={bi}')

        if 'sigmas' in test_data:
            sigmas = test_data['sigmas']
            if torch.is_tensor(sigmas):
                sigmas = sigmas.detach().cpu().numpy()
            sigmas = np.asarray(sigmas).reshape(-1).tolist()
            sigmas = [round(float(s), 4) for s in sigmas]
            extra_parts.append(f'sigmas={sigmas}')

        extra = (' | ' + ' ; '.join(extra_parts)) if extra_parts else ''
        logger.info(
            '{:->4d}--> {:>20s} | PSNR(float): {:<7.4f} dB | PSNR(uint8): {:<7.4f} dB{}'.format(
                idx, image_name_ext, current_psnr, current_psnr_uint, extra
            )
        )

    if idx == 0:
        logger.warning('Validation skipped because test_loader is empty.')
        return

    avg_psnr /= idx
    avg_psnr_uint /= idx
    logger.info(
        '<epoch:{:3d}, iter:{:8,d}, Average PSNR(float): {:<.4f} dB, Average PSNR(uint8): {:<.4f} dB>\n'.format(
            epoch, current_step, avg_psnr, avg_psnr_uint
        )
    )



def main(json_path='options/train_hsi_dncnn.json'):
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, default=json_path, help='Path to option JSON file.')
    args = parser.parse_args()

    opt = option.parse(args.opt, is_train=True)
    util.mkdirs((path for key, path in opt['path'].items() if 'pretrained' not in key))

    init_iter, init_path_G = option.find_last_checkpoint(opt['path']['models'], net_type='G')
    opt['path']['pretrained_netG'] = init_path_G
    current_step = init_iter
    border = 0

    option.save(opt)
    opt = option.dict_to_nonedict(opt)

    logger_name = 'train'
    utils_logger.logger_info(logger_name, os.path.join(opt['path']['log'], logger_name + '.log'))
    logger = logging.getLogger(logger_name)
    logger.info(option.dict2str(opt))

    crash_log_path = os.path.join(opt['path']['log'], 'train_crash.log')

    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    logger.info('Random seed: {}'.format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    dataset_type = opt['datasets']['train']['dataset_type']
    train_loader = None
    test_loader = None

    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = define_Dataset(dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt['dataloader_batch_size']))
            logger.info('Number of train samples: {:,d}, iters per epoch: {:,d}'.format(len(train_set), train_size))
            train_loader = DataLoader(
                train_set,
                batch_size=dataset_opt['dataloader_batch_size'],
                shuffle=dataset_opt['dataloader_shuffle'],
                num_workers=dataset_opt['dataloader_num_workers'],
                drop_last=True,
                pin_memory=True,
            )
        elif phase == 'test':
            test_set = define_Dataset(dataset_opt)
            logger.info('Number of validation samples: {:,d}'.format(len(test_set)))
            test_loader = DataLoader(
                test_set,
                batch_size=1,
                shuffle=False,
                num_workers=1,
                drop_last=False,
                pin_memory=True,
            )
        else:
            raise NotImplementedError('Phase [%s] is not recognized.' % phase)

    model = define_Model(opt)

    if opt['merge_bn'] and current_step > opt['merge_bn_startpoint']:
        logger.info('^_^ -----merging bnorm----- ^_^')
        model.merge_bnorm_test()

    logger.info(model.info_network())
    model.init_train()
    logger.info(model.info_params())

    try:
        for epoch in range(1000000):
            for i, train_data in enumerate(train_loader):
                current_step += 1

                if dataset_type == 'dnpatch' and current_step % 20000 == 0:
                    train_loader.dataset.update_data()

                try:
                    model.update_learning_rate(current_step)
                    model.feed_data(train_data)
                    model.optimize_parameters(current_step)
                except Exception as exc:
                    write_crash_dump(
                        crash_log_path,
                        stage='train_step',
                        exc=exc,
                        epoch=epoch,
                        iteration_in_epoch=i,
                        current_step=current_step,
                        train_batch_summary=summarize_batch(train_data),
                    )
                    raise

                if opt['merge_bn'] and opt['merge_bn_startpoint'] == current_step:
                    logger.info('^_^ -----merging bnorm----- ^_^')
                    model.merge_bnorm_train()
                    model.print_network()

                if current_step % opt['train']['checkpoint_print'] == 0:
                    logs = model.current_log()
                    message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> '.format(
                        epoch, current_step, model.current_learning_rate()
                    )
                    for k, v in logs.items():
                        message += '{:s}: {:.3e} '.format(k, v)
                    logger.info(message)

                if current_step % opt['train']['checkpoint_save'] == 0:
                    logger.info('Saving the model.')
                    model.save(current_step)

                if test_loader is not None and current_step % opt['train']['checkpoint_test'] == 0:
                    try:
                        validate(model, test_loader, opt, current_step, epoch, logger, border=border)
                    except Exception as exc:
                        write_crash_dump(
                            crash_log_path,
                            stage='validation',
                            exc=exc,
                            epoch=epoch,
                            current_step=current_step,
                            train_batch_summary=summarize_batch(train_data),
                        )
                        raise

    except KeyboardInterrupt:
        logger.warning('Training interrupted by user. Saving latest model before exit.')
        model.save('latest')
        raise
    except Exception as exc:
        write_crash_dump(
            crash_log_path,
            stage='fatal',
            exc=exc,
            current_step=current_step,
        )
        raise

    logger.info('Saving the final model.')
    model.save('latest')
    logger.info('End of training.')


if __name__ == '__main__':
    main()
