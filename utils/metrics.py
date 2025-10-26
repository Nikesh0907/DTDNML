#!/usr/bin/env python3
"""
Robust evaluation metrics for hyperspectral reconstruction.

Functions assume inputs are numpy arrays with values in [0, 1]. Shapes accepted:
- (H, W, C)
- (C, H, W)
- (B, C, H, W) or (B, H, W, C) – only the first sample is used.

Metrics provided:
- rmse, psnr, sam (degrees), ergas, ssim (band-average), uiqi
- evaluate_pair: convenience wrapper computing all of the above
"""

from __future__ import annotations

import numpy as np
from typing import Tuple, Dict

try:
    # Prefer modern skimage API if available
    from skimage.metrics import structural_similarity as _ssim_metric
except Exception:
    _ssim_metric = None


def _ensure_chw_numpy(arr: np.ndarray) -> np.ndarray:
    """Ensure numpy array shape is (C, H, W). Accepts (B,C,H,W), (C,H,W), or (H,W,C).
    Uses conservative heuristics to distinguish CHW vs HWC.
    """
    a = np.asarray(arr)
    if a.ndim == 4:
        # Prefer NCHW if channel dim small (<= 64); else if NHWC with last dim small
        if a.shape[1] <= 64:
            a = a[0]  # NCHW -> CHW
        elif a.shape[-1] <= 64:
            a = np.transpose(a[0], (2, 0, 1))  # NHWC -> CHW
        else:
            a = a[0]
    if a.ndim != 3:
        raise ValueError(f"Expected 3D or 4D array, got shape {a.shape}")
    # At this point a is 3D. Decide if CHW or HWC.
    C, H, W = a.shape[0], a.shape[1], a.shape[2]
    if C <= 64 and H >= 8 and W >= 8:
        # Likely CHW already
        return a
    if a.shape[-1] <= 64 and a.shape[0] >= 8 and a.shape[1] >= 8:
        # HWC -> CHW
        return np.transpose(a, (2, 0, 1))
    # Fallback: assume CHW
    return a


def rmse(img_ref: np.ndarray, img_rec: np.ndarray) -> float:
    x = np.asarray(img_ref, dtype=np.float32)
    y = np.asarray(img_rec, dtype=np.float32)
    return float(np.sqrt(np.mean((x - y) ** 2)))


def psnr(img_ref: np.ndarray, img_rec: np.ndarray, data_range: float = 1.0) -> float:
    x = np.asarray(img_ref, dtype=np.float32)
    y = np.asarray(img_rec, dtype=np.float32)
    mse = float(np.mean((x - y) ** 2))
    if mse <= 0.0:
        return float('inf')
    return float(20.0 * np.log10(data_range) - 10.0 * np.log10(mse))


def sam(img_ref: np.ndarray, img_rec: np.ndarray, eps: float = 1e-12, as_degrees: bool = True) -> float:
    """Spectral Angle Mapper (per-pixel angle between spectra), averaged over pixels."""
    x = _ensure_chw_numpy(img_ref).astype(np.float64)
    y = _ensure_chw_numpy(img_rec).astype(np.float64)
    if x.shape != y.shape:
        raise ValueError(f"Shape mismatch in SAM: {x.shape} vs {y.shape}")
    C, H, W = x.shape
    xf = x.reshape(C, -1)
    yf = y.reshape(C, -1)
    nom = np.sum(xf * yf, axis=0)
    denom = np.linalg.norm(xf, axis=0) * np.linalg.norm(yf, axis=0)
    cosang = np.clip(nom / np.maximum(denom, eps), -1.0, 1.0)
    ang = np.arccos(cosang)
    if as_degrees:
        ang = ang / np.pi * 180.0
    # ignore non-finite
    ang = ang[np.isfinite(ang)]
    return float(np.mean(ang)) if ang.size else float('nan')


def ergas(img_ref: np.ndarray, img_rec: np.ndarray, ratio: int) -> float:
    """ERGAS = 100 * (1/ratio) * sqrt( mean_b( (RMSE_b / mean_b)^2 ) ). Lower is better."""
    x = _ensure_chw_numpy(img_ref).astype(np.float64)
    y = _ensure_chw_numpy(img_rec).astype(np.float64)
    if x.shape != y.shape:
        raise ValueError(f"Shape mismatch in ERGAS: {x.shape} vs {y.shape}")
    rmse_b = np.sqrt(np.mean((x - y) ** 2, axis=(1, 2)))
    mean_b = np.mean(x, axis=(1, 2))
    val = np.mean((rmse_b / (mean_b + 1e-12)) ** 2)
    return float(100.0 * (1.0 / float(ratio)) * np.sqrt(val))


def ssim_band_mean(img_ref: np.ndarray, img_rec: np.ndarray, data_range: float = 1.0) -> float:
    """Mean SSIM across bands using skimage.metrics.structural_similarity."""
    x = _ensure_chw_numpy(img_ref).astype(np.float32)
    y = _ensure_chw_numpy(img_rec).astype(np.float32)
    C = x.shape[0]
    vals = []
    for c in range(C):
        if _ssim_metric is not None:
            vals.append(float(_ssim_metric(x[c], y[c], data_range=data_range)))
        else:
            # Very old skimage fallback (deprecated API in measure)
            from skimage import measure as _measure
            vals.append(float(_measure.compare_ssim(x[c], y[c], data_range=data_range)))
    return float(np.mean(vals)) if vals else float('nan')


def uiqi(img_ref: np.ndarray, img_rec: np.ndarray) -> float:
    """Universal Image Quality Index averaged across bands (simple implementation)."""
    x = _ensure_chw_numpy(img_ref).astype(np.float64)
    y = _ensure_chw_numpy(img_rec).astype(np.float64)
    C = x.shape[0]
    vals = []
    for c in range(C):
        a = x[c].ravel(); b = y[c].ravel()
        mx = a.mean(); my = b.mean()
        vx = ((a - mx) ** 2).mean(); vy = ((b - my) ** 2).mean()
        cov = ((a - mx) * (b - my)).mean()
        denom = vx + vy + 1e-12
        vals.append((2.0 * cov) / denom)
    return float(np.mean(vals)) if vals else float('nan')


def evaluate_pair(gt: np.ndarray, rec: np.ndarray, scale_factor: int) -> Dict[str, float]:
    gt = np.clip(np.asarray(gt, dtype=np.float32), 0.0, 1.0)
    rec = np.clip(np.asarray(rec, dtype=np.float32), 0.0, 1.0)
    metrics = {
        'RMSE': rmse(gt, rec),
        'PSNR': psnr(gt, rec, data_range=1.0),
        'SAM': sam(gt, rec),
        'ERGAS': ergas(gt, rec, ratio=scale_factor),
        'SSIM': ssim_band_mean(gt, rec, data_range=1.0),
        'UIQI': uiqi(gt, rec),
    }
    return metrics
