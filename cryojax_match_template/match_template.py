"""Full-micrograph 2DTM following cisTEM match_template, with cryojax as projector.

Pipeline (cisTEM match_template / TemplateMatchingCore):
  - image: 5-sigma outlier clip -> 1/sqrt(PSD) whitening (max=1) -> zero DC ->
    unit mean-square -> Fourier-crop to search pixel size -> swap quadrants ->
    zero DC -> /sqrt(sumsq/N)
  - template: projection (CTF applied) x same whitening curve -> subtract edge
    mean -> unit variance over the padded (full-image) box -> centered pad
  - correlation: conj(template_fft) * image_fft -> iFFT
  - output: global mean/std z-score -> MIP; per-pixel avg/std over the search ->
    scaled MIP; best phi/theta/psi/defocus per pixel
"""
import os
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np
import jax
import jax.numpy as jnp
import cryojax.simulator as cxs

from .pose_grid import cistem_pose_grid
from .projector import make_proj_fn
from .whitening import (
    replace_outliers_with_mean, compute_whitening_filter_cistem,
    local_whitened_micrograph_cistem,
)


def fourier_crop(img, out_h, out_w):
    """Downsample by cropping the centered FFT to (out_h, out_w)."""
    H, W = img.shape
    F = np.fft.fftshift(np.fft.fft2(img))
    y0, x0 = H // 2 - out_h // 2, W // 2 - out_w // 2
    Fc = F[y0:y0 + out_h, x0:x0 + out_w]
    out = np.fft.ifft2(np.fft.ifftshift(Fc)).real
    return out.astype(np.float32) * (out_h * out_w) / (H * W)


def preprocess_micrograph(mic, native_px, search_px, whiten_mode="global",
                          whiten_patch=512, whiten_stride=256):
    """Whiten, bin and normalize the micrograph.

    Returns (image_fft, Hf, search_shape):
      image_fft: rFFT of the processed search image (not conjugated)
      Hf: global radial whitening curve (native cyc/px bins), also applied to
          the templates in both whitening modes
    whiten_mode: "global" (as in cisTEM) or "local" (patch-wise, see whitening.py)
    """
    wf_mic, Hf = compute_whitening_filter_cistem(mic)
    if whiten_mode == "local":
        w = local_whitened_micrograph_cistem(mic, patch=whiten_patch,
                                             stride=whiten_stride, sigma=5.0)
    elif whiten_mode == "global":
        F = np.fft.rfft2(replace_outliers_with_mean(mic, sigma=5.0))
        F[0, 0] = 0.0
        w = np.fft.irfft2(F * wf_mic, s=mic.shape).real.astype(np.float32)
    else:
        raise ValueError(f"unknown whiten_mode={whiten_mode!r}")
    w = w - w.mean()
    # cisTEM ReturnSumOfSquares() is a mean square for real-space images
    w = w / (np.sqrt(np.mean(w ** 2)) + 1e-20)

    H, W = mic.shape
    bin_factor = search_px / native_px
    out_h = int(round(H / bin_factor)) & ~1
    out_w = int(round(W / bin_factor)) & ~1
    binned = fourier_crop(w, out_h, out_w)

    binned = np.fft.fftshift(binned)  # SwapRealSpaceQuadrants
    Fb = np.fft.rfft2(binned)
    Fb[0, 0] = 0.0
    sumsq = np.sum(np.abs(np.fft.irfft2(Fb, s=(out_h, out_w))) ** 2)
    Fb = Fb / (np.sqrt(sumsq / (out_h * out_w)) + 1e-20)
    return jnp.asarray(Fb), Hf, (out_h, out_w)


def template_whitening_filter(Hf, native_px, tmpl_box, search_px):
    """Map the radial curve Hf onto the template rFFT grid by physical
    frequency (cyc/A), so the same filter applies at the search pixel size."""
    nbins = len(Hf)
    max_freq = 0.5 * np.sqrt(2.0)
    yy = np.fft.fftfreq(tmpl_box)[:, None] / search_px
    xx = np.fft.rfftfreq(tmpl_box)[None, :] / search_px
    r_A = np.sqrt(yy ** 2 + xx ** 2)
    bin_freq_A = (np.arange(nbins) / (nbins - 1) * max_freq) / native_px
    wf = np.interp(r_A, bin_freq_A, Hf, left=Hf[0], right=0.0)
    return jnp.asarray(wf.astype(np.float32))


def centered_pad_2d(img, out_shape):
    """Embed img at the same physical center as cisTEM ClipInto."""
    H, W = out_shape
    h, w = img.shape
    y0, x0 = (H - h) // 2, (W - w) // 2
    return jnp.zeros((H, W), dtype=img.dtype).at[y0:y0 + h, x0:x0 + w].set(img)


def make_corr_fn(proj, tmpl_wf, img_fft, img_shape, tmpl_box):
    """Returns jitted corr(phi, theta, psi, df1, df2, dfang, Cs, amp) -> (H, W) CCC map."""
    Himg, Wimg = img_shape
    Nt = tmpl_box
    N = float(Himg * Wimg)

    def corr(phi, theta, psi, df1, df2, dfang, Cs, amp):
        t = proj(phi, theta, psi, 0.0, 0.0, df1, df2, dfang, Cs, amp)
        t = jnp.fft.irfft2(jnp.fft.rfft2(t) * tmpl_wf, s=(Nt, Nt))
        # subtract edge mean
        em = (jnp.sum(t[0]) + jnp.sum(t[-1]) + jnp.sum(t[:, 0]) + jnp.sum(t[:, -1])
              - t[0, 0] - t[0, -1] - t[-1, 0] - t[-1, -1]) / (4.0 * Nt - 4.0)
        t = t - em
        # unit variance over the padded box
        mean = jnp.sum(t) / N
        var = jnp.sum(t ** 2) / N - mean ** 2
        t = t / (jnp.sqrt(var) + 1e-20)
        Pf = jnp.fft.rfft2(centered_pad_2d(t, (Himg, Wimg)))
        Pf = Pf.at[0, 0].set(0.0)
        return jnp.fft.irfft2(jnp.conj(Pf) * img_fft, s=(Himg, Wimg))

    return jax.jit(corr)


def match_template(mic, volume, native_px, search_px, df1, df2, astig_angle=0.0,
                   cs=2.7, amp=0.07, voltage_kv=300.0, oop_step=10.0, ip_step=7.5,
                   symmetry="C1", defocus_range=1200.0, defocus_step=200.0,
                   whiten_mode="global", whiten_patch=512, whiten_stride=256,
                   pad_scale=1.0, batch=200, max_orientations=0, verbose=True):
    """Run a full 2DTM search on one micrograph.

    mic: 2D array at native_px. volume: cubic 3D array at search_px.
    Defocus in A, astig_angle in degrees, cs in mm.

    Returns (maps, info). maps has mip, scaled_mip, avg, std, phi, theta, psi,
    defocus (defocus = offset from df1/df2), each (H, W) at search_px.
    """
    t0 = time.perf_counter()
    log = print if verbose else (lambda *a, **k: None)
    mic = np.asarray(mic, dtype=np.float32)
    vg = np.asarray(volume, dtype=np.float32)
    Nt = vg.shape[0]

    img_fft, Hf, (Himg, Wimg) = preprocess_micrograph(
        mic, native_px, search_px, whiten_mode=whiten_mode,
        whiten_patch=whiten_patch, whiten_stride=whiten_stride)
    log(f"search image {(Himg, Wimg)} @ {search_px} A/px [whiten={whiten_mode}]", flush=True)

    vol = cxs.FourierVoxelGridVolume.from_real_voxel_grid(vg, pad_scale=pad_scale)
    cfg = cxs.BasicImageConfig(shape=(Nt, Nt), pixel_size=search_px,
                               voltage_in_kilovolts=voltage_kv, precompute_mode="rfft")
    proj = make_proj_fn(vol, cfg)
    tmpl_wf = template_whitening_filter(Hf, native_px, Nt, search_px)

    poses, sphere, psis, _ = cistem_pose_grid(symmetry, oop_step, ip_step)
    if max_orientations:
        poses = poses[:max_orientations]
    nd = round(defocus_range / defocus_step)
    df_offsets = np.arange(-nd, nd + 1) * defocus_step
    n_combos = len(poses) * len(df_offsets)
    log(f"sphere={len(sphere)} psi={len(psis)} poses={len(poses)} "
        f"defoci={len(df_offsets)} total={n_combos}", flush=True)

    corr = make_corr_fn(proj, tmpl_wf, img_fft, (Himg, Wimg), Nt)

    # Accumulate on device; defoci are scanned so no (D, H, W) stack is built.
    df1s = jnp.asarray(df1 + df_offsets, dtype=jnp.float32)
    df2s = jnp.asarray(df2 + df_offsets, dtype=jnp.float32)
    doffs = jnp.asarray(df_offsets, dtype=jnp.float32)
    dfang, cs, amp = float(astig_angle), float(cs), float(amp)

    def pose_step(acc, pose3):
        phi, theta, psi = pose3[0], pose3[1], pose3[2]

        def defocus_step(carry, values):
            mip, bp, bt, bs, bd, csum, csumsq = carry
            d1, d2, doff = values
            cc = corr(phi, theta, psi, d1, d2, dfang, cs, amp)
            upd = cc > mip
            return (jnp.where(upd, cc, mip), jnp.where(upd, phi, bp),
                    jnp.where(upd, theta, bt), jnp.where(upd, psi, bs),
                    jnp.where(upd, doff, bd), csum + cc, csumsq + cc * cc), None

        return jax.lax.scan(defocus_step, acc, (df1s, df2s, doffs))[0]

    def run_chunk(acc, chunk):
        return jax.lax.scan(lambda c, p: (pose_step(c, p), None), acc, chunk)[0]
    run_chunk = jax.jit(run_chunk, donate_argnums=(0,))

    acc = (jnp.full((Himg, Wimg), -1e30, jnp.float32),
           *(jnp.zeros((Himg, Wimg), dtype=jnp.float32) for _ in range(6)))
    poses_dev = jnp.asarray(poses, dtype=jnp.float32)
    batch = max(1, batch)
    for start in range(0, len(poses), batch):
        stop = min(start + batch, len(poses))
        acc = run_chunk(acc, poses_dev[start:stop])
        acc[0].block_until_ready()
        log(f"  pose {stop}/{len(poses)} ({time.perf_counter() - t0:.0f}s)", flush=True)

    mip, phi_m, theta_m, psi_m, defocus_m, csum, csumsq = (np.asarray(a) for a in acc)
    csum = csum.astype(np.float64)
    csumsq = csumsq.astype(np.float64)

    # global z-score, then local flat-field
    n = float(n_combos)
    valid = csumsq > 1e-10
    total = n * valid.sum()
    gmean = csum[valid].sum() / total
    gstd = np.sqrt(csumsq[valid].sum() / total - gmean ** 2)
    log(f"global CCC mean={gmean:.4f} std={gstd:.4f}", flush=True)

    mip_z = (mip - gmean) / gstd
    sumsqp = (csumsq - 2 * gmean * csum + n * gmean ** 2) / gstd ** 2
    sump = (csum - n * gmean) / gstd
    avg = sump / n
    std = np.sqrt(np.maximum(sumsqp / n - avg ** 2, 0))
    scaled = np.where(valid & (std > 1e-10), (mip_z - avg) / (std + 1e-20), mip_z)

    maps = {"mip": mip_z, "scaled_mip": scaled, "avg": avg, "std": std,
            "phi": phi_m, "theta": theta_m, "psi": psi_m, "defocus": defocus_m}
    maps = {k: v.astype(np.float32) for k, v in maps.items()}
    info = dict(search_shape=(Himg, Wimg), n_sphere_positions=len(sphere),
                n_psi_rotations=len(psis), n_poses=len(poses),
                n_defoci=len(df_offsets), gmean=float(gmean), gstd=float(gstd),
                seconds=time.perf_counter() - t0)
    return maps, info
