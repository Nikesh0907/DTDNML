#!/usr/bin/env python3
"""
Evaluation script: load checkpoint, run inference on dataloader, compute RMSE/PSNR/SAM/ERGAS/SSIM/UIQI, save results.
Usage example:
  python3 test_infer.py --name paviaU_scale_8 --data_name paviaU --mat_name PaviaU --srf_name paviaU --which_epoch latest --scale_factor 8
"""
import os
import argparse
import numpy as np
import scipy.io as sio
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim

from options.train_options import TrainOptions
from data import get_dataloader
from model import create_model


def rmse(img_ref, img_rec):
    return np.sqrt(np.mean((img_ref - img_rec) ** 2))


def psnr(img_ref, img_rec, data_range=1.0):
    mse = np.mean((img_ref - img_rec) ** 2)
    if mse == 0:
        return float('inf')
    return 20.0 * np.log10(data_range) - 10.0 * np.log10(mse)


def sam(img_ref, img_rec):
    f = img_ref.reshape(-1, img_ref.shape[-1])
    g = img_rec.reshape(-1, img_rec.shape[-1])
    dot = np.sum(f * g, axis=1)
    fn = np.linalg.norm(f, axis=1)
    gn = np.linalg.norm(g, axis=1)
    cos = np.clip(dot / (fn * gn + 1e-12), -1.0, 1.0)
    ang = np.arccos(cos)
    return np.mean(ang) * (180.0 / np.pi)


def ergas(img_ref, img_rec, ratio):
    B = img_ref.shape[-1]
    rmse_b = np.sqrt(np.mean((img_ref - img_rec) ** 2, axis=(0, 1)))
    mean_b = np.mean(img_ref, axis=(0, 1))
    val = np.mean((rmse_b / (mean_b + 1e-12)) ** 2)
    return 100.0 * (1.0 / ratio) * np.sqrt(val)


def uiqi(img_ref, img_rec):
    H, W, B = img_ref.shape
    uiqi_list = []
    for b in range(B):
        x = img_ref[..., b].ravel()
        y = img_rec[..., b].ravel()
        mx = x.mean(); my = y.mean()
        vx = ((x - mx) ** 2).mean()
        vy = ((y - my) ** 2).mean()
        cov = ((x - mx) * (y - my)).mean()
        denom = vx + vy + 1e-12
        uiqi_list.append((2.0 * cov) / denom)
    return float(np.mean(uiqi_list))


def evaluate_pair(gt, rec, scale_factor):
    gt = np.clip(gt, 0.0, 1.0).astype(np.float32)
    rec = np.clip(rec, 0.0, 1.0).astype(np.float32)
    metrics = {}
    metrics['RMSE'] = rmse(gt, rec)
    metrics['PSNR'] = psnr(gt, rec, data_range=1.0)
    metrics['SAM'] = sam(gt, rec)
    metrics['ERGAS'] = ergas(gt, rec, scale_factor)
    B = gt.shape[-1]
    ssim_list = []
    for b in range(B):
        try:
            ssim_list.append(ssim(gt[..., b], rec[..., b], data_range=1.0))
        except Exception:
            ssim_list.append(0.0)
    metrics['SSIM'] = float(np.mean(ssim_list))
    metrics['UIQI'] = uiqi(gt, rec)
    return metrics


def main():
    # allow CLI overrides but reuse TrainOptions parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, help='experiment name')
    parser.add_argument('--which_epoch', type=str, default='latest')
    parser.add_argument('--data_name', type=str, help='dataset folder name')
    parser.add_argument('--mat_name', type=str, help='mat base name')
    parser.add_argument('--srf_name', type=str, help='srf base name')
    parser.add_argument('--scale_factor', type=int, help='scale factor (4 or 8)')
    args, _ = parser.parse_known_args()

    opt = TrainOptions().parse()
    # apply CLI overrides if present
    if args.name:
        opt.name = args.name
    if args.which_epoch:
        opt.which_epoch = args.which_epoch
    if args.data_name:
        opt.data_name = args.data_name
    if args.mat_name:
        opt.mat_name = args.mat_name
    if args.srf_name:
        opt.srf_name = args.srf_name
    if args.scale_factor:
        opt.scale_factor = args.scale_factor

    opt.isTrain = False
    opt.batchsize = 1

    dataloader = get_dataloader(opt, isTrain=False)
    model = create_model(opt, dataloader.hsi_channels, dataloader.msi_channels,
                         dataloader.lrhsi_height, dataloader.lrhsi_width,
                         dataloader.sp_matrix, dataloader.sp_range)
    model.setup(opt)
    # attempt to load networks
    try:
        model.load_networks(opt.which_epoch)
    except Exception as e:
        print('Failed to load networks:', e)

    results_dir = os.path.join('./checkpoints', opt.name, 'results_eval')
    os.makedirs(results_dir, exist_ok=True)

    rows = []
    for i, data in enumerate(tqdm(dataloader)):
        model.set_input(data, isTrain=False)
        import torch
        with torch.no_grad():
            if hasattr(model, 'my_forward'):
                model.my_forward(epoch=0)
            else:
                model.forward()
        visuals = model.get_current_visuals()
        key_map = model.get_visual_corresponding_name()
        # expect mapping contains 'real_hhsi'
        vis_key = key_map.get('real_hhsi', None)
        if vis_key is None or vis_key not in visuals:
            print('visual key for reconstructed HSI not found in model; keys:', key_map)
            break
        rec = visuals[vis_key].data.cpu().float().numpy()[0]
        # C x H x W -> H x W x C
        rec = np.transpose(rec, (1, 2, 0))

        # try getting GT from dataloader entry
        if 'hhsi' in data:
            gt = data['hhsi'].cpu().numpy()[0]
            gt = np.transpose(gt, (1, 2, 0))
        else:
            # try to get gt from visuals mapping if provided
            gt_key = key_map.get('gt_hhsi', None)
            if gt_key and gt_key in visuals:
                gt = visuals[gt_key].data.cpu().float().numpy()[0]
                gt = np.transpose(gt, (1, 2, 0))
            else:
                print('Ground truth HSI not found for sample', i)
                break

        metrics = evaluate_pair(gt, rec, opt.scale_factor)
        name = ''.join(data.get('name', [f'img_{i}']))
        metrics['name'] = name
        rows.append(metrics)
        sio.savemat(os.path.join(results_dir, name + '_rec.mat'), {'out': rec.transpose(2, 0, 1)})

    # save CSV
    try:
        import pandas as pd
        df = pd.DataFrame(rows).set_index('name')
        df.to_csv(os.path.join(results_dir, 'metrics_per_image.csv'))
        print('Average metrics:\n', df.mean().to_dict())
    except Exception as e:
        print('Could not write CSV:', e)


if __name__ == '__main__':
    main()
