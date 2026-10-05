#!/usr/bin/env python
"""Compare cryojax-match-template output maps with a cisTEM match_template run.

  python scripts/compare_to_cistem.py --ours results/mic147 \
      --cistem /path/TemplateMatching/<name>_{}_29_3.mrc

`--cistem` is a pattern where {} is replaced by the map name (mip, scaled_mip, ...).
Peaks are picked on the cisTEM scaled MIP and matched to ours within --radius px.
"""
import argparse

import numpy as np
import mrcfile
from scipy import ndimage
from scipy.spatial.transform import Rotation

MAPS = ["mip", "scaled_mip", "avg", "std", "phi", "theta", "psi", "defocus"]


def read(path):
    with mrcfile.open(path, permissive=True) as m:
        return np.array(m.data.squeeze(), dtype=np.float32)


def pick_peaks(img, threshold, min_dist, valid):
    mx = ndimage.maximum_filter(img, size=2 * min_dist + 1)
    ys, xs = np.nonzero((img == mx) & (img > threshold) & valid)
    order = np.argsort(-img[ys, xs])
    return ys[order], xs[order]


def geodesic_deg(a, b):
    """Angle between cisTEM ZYZ Euler triples (N, 3), degrees."""
    ra = Rotation.from_euler("ZYZ", a, degrees=True)
    rb = Rotation.from_euler("ZYZ", b, degrees=True)
    return np.rad2deg((ra.inv() * rb).magnitude())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", required=True, help="our --out-prefix")
    ap.add_argument("--cistem", required=True, help="cisTEM map path with {} for map name")
    ap.add_argument("--thresholds", default="7.0,6.5")
    ap.add_argument("--radius", type=int, default=2, help="peak match radius (px)")
    ap.add_argument("--min-dist", type=int, default=10, help="peak picking min distance (px)")
    ap.add_argument("--edge", type=int, default=200, help="ignore this many px at edges")
    args = ap.parse_args()

    ours = {k: read(f"{args.ours}_{k}.mrc") for k in MAPS}
    cis = {k: read(args.cistem.format(k)) for k in MAPS}
    if ours["mip"].shape != cis["mip"].shape:
        # our maps are at the search pixel size; cisTEM upsamples back to the input
        zoom = [c / o for c, o in zip(cis["mip"].shape, ours["mip"].shape)]
        print(f"resampling ours {ours['mip'].shape} -> {cis['mip'].shape} (zoom {zoom[0]:.3f})")
        ours = {k: ndimage.zoom(v, zoom, order=1 if k in ("mip", "scaled_mip", "avg", "std") else 0)
                for k, v in ours.items()}
        ours = {k: v[:cis["mip"].shape[0], :cis["mip"].shape[1]] for k, v in ours.items()}

    valid = cis["scaled_mip"] != 0
    valid[:args.edge] = valid[-args.edge:] = False
    valid[:, :args.edge] = valid[:, -args.edge:] = False

    print(f"\nvalid pixels: {valid.sum()}")
    print(f"{'map':<11}{'cisTEM mean/std':>20}{'ours mean/std':>20}{'pearson r':>11}")
    for k in ["mip", "scaled_mip", "avg", "std"]:
        a, b = cis[k][valid], ours[k][valid]
        r = np.corrcoef(a, b)[0, 1]
        print(f"{k:<11}{a.mean():>10.3f}/{a.std():<9.3f}{b.mean():>10.3f}/{b.std():<9.3f}{r:>11.3f}")

    ours_max = ndimage.maximum_filter(ours["scaled_mip"], size=2 * args.radius + 1)
    for thr in [float(t) for t in args.thresholds.split(",")]:
        ys, xs = pick_peaks(cis["scaled_mip"], thr, args.min_dist, valid)
        n = len(ys)
        if n == 0:
            print(f"\n> {thr} sigma: no cisTEM peaks")
            continue
        found = ours_max[ys, xs] > thr
        # pose/defocus from our map at the cisTEM peak pixel
        cp = np.stack([cis[k][ys, xs] for k in ("phi", "theta", "psi")], 1)
        op = np.stack([ours[k][ys, xs] for k in ("phi", "theta", "psi")], 1)
        ang = geodesic_deg(cp, op)
        ddf = np.abs(cis["defocus"][ys, xs] - ours["defocus"][ys, xs])
        oy, ox = pick_peaks(ours["scaled_mip"], thr, args.min_dist, valid)
        print(f"\n> {thr} sigma: cisTEM {n} peaks, ours {len(oy)} peaks")
        print(f"  cisTEM peaks also > {thr} in ours (within {args.radius}px): "
              f"{found.sum()}/{n} ({100 * found.mean():.0f}%)")
        print(f"  pose difference at cisTEM peaks: median {np.median(ang):.1f} deg, "
              f"<10 deg {100 * np.mean(ang < 10):.0f}%")
        print(f"  defocus difference: median {np.median(ddf):.0f} A, max {ddf.max():.0f} A")
        print(f"  scaled MIP at cisTEM peaks: cisTEM median {np.median(cis['scaled_mip'][ys, xs]):.2f}, "
              f"ours median {np.median(ours['scaled_mip'][ys, xs]):.2f}")


if __name__ == "__main__":
    main()
