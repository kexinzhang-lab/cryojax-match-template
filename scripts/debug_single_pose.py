#!/usr/bin/env python
"""Single-pose check against a cisTEM run.

Takes the top peaks of the cisTEM scaled MIP, projects the template at the pose
(and defocus) cisTEM found there, and checks that our CCC map for that one
template peaks at the same pixel. This tests whitening, CTF, normalization and
Euler conventions without the orientation search.

  python scripts/debug_single_pose.py --mic mic.mrc --volume tmpl.mrc \
      --native-px 1.06 --search-px 1.06 --df1 6308 --df2 5297 --astig-angle -66.06 \
      --cistem /path/<name>_{}_29_3.mrc
"""
import argparse

import numpy as np
import mrcfile
import cryojax.simulator as cxs

from cryojax_match_template.match_template import (
    preprocess_micrograph, make_corr_fn, template_whitening_filter)
from cryojax_match_template.projector import make_proj_fn


def read(path):
    with mrcfile.open(path, permissive=True) as m:
        return np.array(m.data.squeeze(), dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mic", required=True)
    ap.add_argument("--volume", required=True)
    ap.add_argument("--native-px", type=float, required=True)
    ap.add_argument("--search-px", type=float, required=True)
    ap.add_argument("--df1", type=float, required=True)
    ap.add_argument("--df2", type=float, required=True)
    ap.add_argument("--astig-angle", type=float, default=0.0)
    ap.add_argument("--cs", type=float, default=2.7)
    ap.add_argument("--amp", type=float, default=0.07)
    ap.add_argument("--voltage", type=float, default=300.0)
    ap.add_argument("--cistem", required=True, help="cisTEM map path with {} for map name")
    ap.add_argument("--n-peaks", type=int, default=5)
    ap.add_argument("--whiten-mode", default="global")
    args = ap.parse_args()

    smip = read(args.cistem.format("scaled_mip"))
    phi, theta, psi, dfm = (read(args.cistem.format(k)) for k in ("phi", "theta", "psi", "defocus"))
    mic = read(args.mic)
    vg = read(args.volume)
    Nt = vg.shape[0]

    img_fft, Hf, shape = preprocess_micrograph(mic, args.native_px, args.search_px,
                                               whiten_mode=args.whiten_mode)
    vol = cxs.FourierVoxelGridVolume.from_real_voxel_grid(vg, pad_scale=1.0)
    cfg = cxs.BasicImageConfig(shape=(Nt, Nt), pixel_size=args.search_px,
                               voltage_in_kilovolts=args.voltage, precompute_mode="rfft")
    corr = make_corr_fn(make_proj_fn(vol, cfg),
                        template_whitening_filter(Hf, args.native_px, Nt, args.search_px),
                        img_fft, shape, Nt)
    scale = smip.shape[0] / shape[0]  # cisTEM map px per our px

    flat = np.argsort(-smip, axis=None)
    taken = []
    for idx in flat:
        y, x = np.unravel_index(idx, smip.shape)
        if any(abs(y - a) < 50 and abs(x - b) < 50 for a, b in taken):
            continue
        taken.append((y, x))
        P, T, S, D = (float(m[y, x]) for m in (phi, theta, psi, dfm))
        cc = np.asarray(corr(P, T, S, args.df1 + D, args.df2 + D, args.astig_angle,
                             args.cs, args.amp))
        z = (cc - cc.mean()) / cc.std()
        oy, ox = np.unravel_index(np.argmax(z), z.shape)
        ty, tx = int(round(y / scale)), int(round(x / scale))
        print(f"cisTEM peak ({y},{x}) smip={smip[y, x]:.2f} pose=({P:.1f},{T:.1f},{S:.1f}) df={D:+.0f}"
              f" | ours: argmax ({oy * scale:.0f},{ox * scale:.0f}) z={z[oy, ox]:.2f},"
              f" z at cisTEM pixel={z[ty, tx]:.2f}, offset={np.hypot(oy - ty, ox - tx) * scale:.1f}px")
        if len(taken) >= args.n_peaks:
            break


if __name__ == "__main__":
    main()
