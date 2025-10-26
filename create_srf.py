#!/usr/bin/env python3
"""
Create a simple spectral response .xls file for a dataset.
Writes an .xls with a dummy first column (wavelengths) and subsequent columns as spectral response.
Usage:
  python3 create_srf.py --dataset paviaU --mat PaviaU --msi_bands 3 --srf_name paviaU
This will write ./paviaU/paviaU.xls
"""
import os
import scipy.io as sio
import numpy as np
import xlwt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='dataset folder under repo root')
parser.add_argument('--mat', required=True, help='mat filename base (without .mat)')
parser.add_argument('--msi_bands', type=int, default=3, help='number of MSI bands to simulate')
parser.add_argument('--srf_name', required=True, help='output srf base name (without .xls)')
parser.add_argument('--var_name', default=None, help='optional variable name inside .mat (e.g., img or paviaU); if omitted, auto-detect a 3D HxWxC array')
args = parser.parse_args()

repo = os.getcwd()
mat_path = os.path.join(repo, args.dataset, args.mat + '.mat')
if not os.path.exists(mat_path):
    raise SystemExit('Mat file not found: %s' % mat_path)

def find_hsi_array(mat_dict, preferred_key=None):
    # 1) preferred key
    if preferred_key and preferred_key in mat_dict:
        arr = np.asarray(mat_dict[preferred_key])
        if arr.ndim == 3:
            return arr
    # 2) common key 'img' (case-insensitive)
    for k in list(mat_dict.keys()):
        if k.lower() == 'img':
            arr = np.asarray(mat_dict[k])
            if arr.ndim == 3:
                return arr
    # 3) any top-level 3D ndarray
    for k, v in mat_dict.items():
        if k.startswith('__'):
            continue
        try:
            a = np.asarray(v)
            if a.ndim == 3:
                return a
        except Exception:
            pass
    # 4) search inside a top-level struct for a 3D field
    for k, v in mat_dict.items():
        if k.startswith('__'):
            continue
        obj = v
        # MATLAB struct via scipy may be numpy.void or have _fieldnames
        try:
            # fieldnames path
            if hasattr(obj, '_fieldnames'):
                for f in obj._fieldnames:
                    a = np.asarray(getattr(obj, f))
                    if a.ndim == 3:
                        return a
            # numpy.void with dtype names
            if isinstance(obj, np.void) and getattr(obj, 'dtype', None) and obj.dtype.names:
                for f in obj.dtype.names:
                    a = np.asarray(obj[f])
                    if a.ndim == 3:
                        return a
        except Exception:
            pass
    return None

m = sio.loadmat(mat_path, struct_as_record=False, squeeze_me=True)
img = find_hsi_array(m, args.var_name)
if img is None:
    raise SystemExit('Could not auto-detect a 3D HxWxC array in mat. Provide --var_name to specify the variable to use.')

HSI_bands = img.shape[2]
msi_bands = args.msi_bands

# build simple block SRF
sp = np.zeros((HSI_bands, msi_bands), dtype=np.float32)
bands_per = HSI_bands // msi_bands
for j in range(msi_bands):
    start = j * bands_per
    end = HSI_bands if j == msi_bands - 1 else (j + 1) * bands_per
    sp[start:end, j] = 1.0
# normalize columns
col_sums = sp.sum(axis=0, keepdims=True)
col_sums[col_sums == 0] = 1.0
sp = sp / col_sums

# write xls
out_path = os.path.join(repo, args.dataset, args.srf_name + '.xls')
wb = xlwt.Workbook()
ws = wb.add_sheet('sheet1')
# first column: wavelengths (dummy)
for i in range(HSI_bands):
    ws.write(i, 0, float(i + 1))
for j in range(msi_bands):
    for i in range(HSI_bands):
        ws.write(i, j + 1, float(sp[i, j]))
wb.save(out_path)
print('Saved SRF to', out_path)
