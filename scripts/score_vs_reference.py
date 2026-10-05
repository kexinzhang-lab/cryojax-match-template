#!/usr/bin/env python
"""Score 2DTM result maps against a reference particle list from a cisTEM project.

The reference is the peak list of one cisTEM template-match result
(TEMPLATE_MATCH_PEAK_LIST_<id>, positions in A), typically a finer search.

  python scripts/score_vs_reference.py --ref-db project.db --ref-id 84 --pixel-size 1.06 \
      "cryojax=results/mic147_{}.mrc" "cisTEM=/path/<name>_{}_29_3.mrc"

Each map argument is label=path-pattern, with {} replaced by the map name.
"""
import argparse
import sqlite3

import numpy as np
import mrcfile
from scipy import ndimage
from scipy.spatial.transform import Rotation


def read(path):
    with mrcfile.open(path, permissive=True) as m:
        return np.array(m.data.squeeze(), dtype=np.float32)


def geodesic_deg(a, b):
    ra = Rotation.from_euler("ZYZ", a, degrees=True)
    rb = Rotation.from_euler("ZYZ", b, degrees=True)
    return np.rad2deg((ra.inv() * rb).magnitude())


def pick_peaks(smip, edge, min_dist):
    v = smip.copy()
    v[:edge] = v[-edge:] = 0
    v[:, :edge] = v[:, -edge:] = 0
    mx = ndimage.maximum_filter(v, size=2 * min_dist + 1)
    ys, xs = np.nonzero((v == mx) & (v > 0))
    o = np.argsort(-v[ys, xs])
    return ys[o], xs[o], v[ys[o], xs[o]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-db", required=True)
    ap.add_argument("--ref-id", type=int, required=True, help="TEMPLATE_MATCH_ID of the reference")
    ap.add_argument("--pixel-size", type=float, required=True)
    ap.add_argument("--edge", type=int, default=192, help="ignore particles/peaks this close to the edge (px)")
    ap.add_argument("--match-radius", type=float, default=10, help="px")
    ap.add_argument("--min-dist", type=int, default=10, help="peak picking min distance (px)")
    ap.add_argument("--top-n", default="85,200,500,1000")
    ap.add_argument("--thresholds", default="7.85,7.0")
    ap.add_argument("maps", nargs="+", help="label=pattern")
    args = ap.parse_args()

    ref = np.array(sqlite3.connect(args.ref_db).execute(
        f"select X_POSITION, Y_POSITION, PHI, THETA, PSI from TEMPLATE_MATCH_PEAK_LIST_{args.ref_id}"
    ).fetchall())
    rx = np.round(ref[:, 0] / args.pixel_size).astype(int)
    ry = np.round(ref[:, 1] / args.pixel_size).astype(int)

    first = True
    for item in args.maps:
        label, pattern = item.split("=", 1)
        m = {k: read(pattern.format(k)) for k in ("scaled_mip", "phi", "theta", "psi")}
        H, W = m["scaled_mip"].shape
        e = args.edge
        inside = (rx >= e) & (rx < W - e) & (ry >= e) & (ry < H - e)
        if first:
            print(f"reference: {len(ref)} particles, {inside.sum()} away from the edge\n")
            tops = [int(t) for t in args.top_n.split(",")]
            thrs = [float(t) for t in args.thresholds.split(",")]
            print(f"{'':<20}" + "".join(f"{'rec@' + str(t):>9}" for t in tops)
                  + "".join(f"{'>' + str(t) + ' n/prec/rec':>21}" for t in thrs)
                  + f"{'pose err med':>14}{'<10deg':>8}{'smip@ref':>10}")
            first = False

        ys, xs, vals = pick_peaks(m["scaled_mip"], e, args.min_dist)
        rank = np.full(len(ref), np.iinfo(np.int64).max)
        for i in np.nonzero(inside)[0]:
            j = np.nonzero(np.hypot(ys - ry[i], xs - rx[i]) <= args.match_radius)[0]
            if len(j):
                rank[i] = j[0]
        recall = lambda n: np.mean(rank[inside] < n)

        cols = "".join(f"{recall(t):>9.2f}" for t in tops)
        for t in thrs:
            n = int(np.sum(vals > t))
            hit = sum(np.any(np.hypot(ry[inside] - y, rx[inside] - x) <= args.match_radius)
                      for y, x in zip(ys[:n], xs[:n]))
            cols += f"{n:>9} / {hit / max(n, 1):.2f} / {recall(n):.2f}"
        found = np.nonzero(inside & (rank < len(ys)))[0]
        py, px = ys[rank[found]], xs[rank[found]]
        pose = np.stack([m[k][py, px] for k in ("phi", "theta", "psi")], 1)
        err = geodesic_deg(pose, ref[found][:, 2:5])
        print(f"{label:<20}{cols}{np.median(err):>13.1f}°{np.mean(err < 10):>8.2f}"
              f"{np.median(m['scaled_mip'][ry[inside], rx[inside]]):>10.2f}")


if __name__ == "__main__":
    main()
