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
from options.train_options import TrainOptions
from data import get_dataloader
from model import create_model
from utils.metrics import evaluate_pair


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
    # attempt to load networks; if 'latest' not present, fallback to highest epoch found
    def _try_load(which):
        try:
            model.load_networks(which)
            return True
        except Exception as e:
            print('Failed to load networks:', e)
            return False

    loaded = _try_load(opt.which_epoch)
    if not loaded:
        # scan checkpoint dir for numbered epochs and pick the max
        ckpt_dir = os.path.join('./checkpoints', opt.name)
        if os.path.isdir(ckpt_dir):
            import re
            epochs = []
            for fn in os.listdir(ckpt_dir):
                m = re.match(r'^(\d+)_net_.*\\.pth$', fn)
                if not m:
                    m = re.match(r'^(\d+)_net_.*\.pth$', fn)
                if m:
                    try:
                        epochs.append(int(m.group(1)))
                    except Exception:
                        pass
            if epochs:
                best = str(max(epochs))
                print('Auto-selecting checkpoint epoch', best)
                _try_load(best)

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
