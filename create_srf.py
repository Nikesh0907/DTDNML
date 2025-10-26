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
args = parser.parse_args()

repo = os.getcwd()
mat_path = os.path.join(repo, args.dataset, args.mat + '.mat')
if not os.path.exists(mat_path):
    raise SystemExit('Mat file not found: %s' % mat_path)

m = sio.loadmat(mat_path)
# expect variable 'img' in mat
if 'img' not in m:
    # try uppercase variants
    found = None
    for k in m.keys():
        if k.lower() == 'img':
            found = k
            break
    if found is None:
        raise SystemExit('Mat file does not contain variable named "img"')
    img = m[found]
else:
    img = m['img']

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
