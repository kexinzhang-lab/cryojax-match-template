"""Whitening, following cisTEM match_template.

1. Replace outliers with the mean (5 sigma)
2. FFT, zero DC
3. 1D radial power spectrum -> filter = 1/sqrt(PSD), normalized to max 1
4. Apply to image, zero DC again, normalize

The template side uses projection_filter = CTF * whitening filter.
"""
import numpy as np


def replace_outliers_with_mean(img, sigma=5.0):
    """Replace pixels beyond sigma*std from the mean with the mean."""
    m = img.mean()
    s = img.std()
    out = img.copy()
    out[np.abs(img - m) > sigma * s] = m
    return out


def compute_whitening_filter_cistem(mic):
    """Radial whitening filter for a micrograph.

    Returns:
        wf_mic: filter on the micrograph rFFT grid, (H, W//2+1)
        Hf: 1D radial filter (nbins,), bins spanning 0..0.5*sqrt(2) cyc/px
    """
    H, W = mic.shape
    F = np.fft.rfft2(replace_outliers_with_mean(mic, sigma=5.0))
    F[0, 0] = 0.0

    P = np.abs(F) ** 2
    yy = np.fft.fftfreq(H).reshape(H, 1)
    xx = np.fft.rfftfreq(W).reshape(1, W // 2 + 1)
    rr = np.sqrt(yy ** 2 + xx ** 2)

    # cisTEM: int((logical_x_dimension / 2 + 1) * sqrt(2) + 1) bins over 0..0.5*sqrt(2)
    nbins = int((min(H, W) / 2.0 + 1.0) * np.sqrt(2.0) + 1.0)
    max_freq = 0.5 * np.sqrt(2.0)
    r_idx = np.clip((rr / max_freq * (nbins - 1)).astype(int), 0, nbins - 1)

    sumP = np.bincount(r_idx.ravel(), weights=P.ravel(), minlength=nbins)
    cntP = np.bincount(r_idx.ravel(), minlength=nbins).clip(1)
    Pr = sumP / cntP

    Hf = np.zeros(nbins, dtype=np.float64)
    Hf[1:] = 1.0 / np.sqrt(Pr[1:].clip(1e-30))
    if Hf.max() > 0:
        Hf = Hf / Hf.max()

    return Hf[r_idx].astype(np.float32), Hf.astype(np.float32)


def _radial_bins(patch):
    yy = np.fft.fftfreq(patch).reshape(patch, 1)
    xx = np.fft.rfftfreq(patch).reshape(1, patch // 2 + 1)
    rr = np.sqrt(yy ** 2 + xx ** 2)
    nbins = int((patch / 2.0 + 1.0) * np.sqrt(2.0) + 1.0)
    max_freq = 0.5 * np.sqrt(2.0)
    r_idx = np.clip((rr / max_freq * (nbins - 1)).astype(int), 0, nbins - 1)
    cnt = np.bincount(r_idx.ravel(), minlength=nbins).clip(1)
    return r_idx, cnt, nbins


def _patch_starts(n, patch, stride):
    if n <= patch:
        return [0]
    xs = list(range(0, n - patch + 1, stride))
    if xs[-1] != n - patch:
        xs.append(n - patch)
    return xs


def local_whitened_micrograph_cistem(mic, patch=512, stride=256, sigma=5.0,
                                     win_floor=1e-3, psd_smooth_bins=7,
                                     psd_floor_fraction=1e-6):
    """Patch-wise (spatially adaptive) whitening. Not part of cisTEM.

    Each overlapping patch gets its own radial 1/sqrt(PSD) curve; patches are
    recombined with a Hann window. Since the filter keeps each patch's absolute
    noise scale, this both flattens the local spectrum and equalizes local
    amplitude (e.g. thick ice / vacuole regions stop dominating the background).

    Returns the whitened micrograph (H, W), not unit-normalized.
    """
    H, W = mic.shape
    clean = replace_outliers_with_mean(mic, sigma=sigma)
    patch = min(patch, H, W)
    stride = max(1, min(stride, patch))
    if psd_smooth_bins < 1 or psd_smooth_bins % 2 == 0:
        raise ValueError("psd_smooth_bins must be a positive odd integer")
    if psd_floor_fraction <= 0:
        raise ValueError("psd_floor_fraction must be positive")

    r_idx, cnt, nbins = _radial_bins(patch)
    win1 = np.hanning(patch)
    win = np.outer(win1, win1).astype(np.float64) + win_floor
    radius = psd_smooth_bins // 2
    kernel = np.full(psd_smooth_bins, 1.0 / psd_smooth_bins)

    out = np.zeros((H, W), dtype=np.float64)
    wsum = np.zeros((H, W), dtype=np.float64)
    for y0 in _patch_starts(H, patch, stride):
        for x0 in _patch_starts(W, patch, stride):
            p = clean[y0:y0 + patch, x0:x0 + patch].astype(np.float64)
            p = p - p.mean()
            F = np.fft.rfft2(p)
            F[0, 0] = 0.0
            P = np.abs(F) ** 2
            Pr = np.bincount(r_idx.ravel(), weights=P.ravel(), minlength=nbins) / cnt
            Pr = np.convolve(np.pad(Pr, (radius, radius), mode="edge"),
                             kernel, mode="valid")
            positive = Pr[1:][Pr[1:] > 0]
            reference_power = np.median(positive) if positive.size else 1.0
            Hf = np.zeros(nbins, dtype=np.float64)
            Hf[1:] = 1.0 / np.sqrt(np.maximum(Pr[1:], reference_power * psd_floor_fraction))
            pw = np.fft.irfft2(F * Hf[r_idx], s=(patch, patch))
            out[y0:y0 + patch, x0:x0 + patch] += pw * win
            wsum[y0:y0 + patch, x0:x0 + patch] += win
    out /= wsum.clip(1e-9)
    return out.astype(np.float32)
