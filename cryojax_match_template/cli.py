import argparse
import hashlib
from pathlib import Path

import numpy as np
import mrcfile

from .match_template import match_template


def file_sha256(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(description="cisTEM-style 2DTM with cryojax")
    ap.add_argument("--mic", required=True, help="micrograph MRC (native pixel size)")
    ap.add_argument("--volume", required=True, help="template volume MRC at --search-px")
    ap.add_argument("--native-px", type=float, required=True, help="micrograph pixel size (A)")
    ap.add_argument("--search-px", type=float, required=True,
                    help="search pixel size (A); micrograph is Fourier-cropped to it")
    ap.add_argument("--df1", type=float, required=True, help="defocus 1 (A)")
    ap.add_argument("--df2", type=float, required=True, help="defocus 2 (A)")
    ap.add_argument("--astig-angle", type=float, default=0.0, help="astigmatism angle (deg)")
    ap.add_argument("--cs", type=float, default=2.7, help="spherical aberration (mm)")
    ap.add_argument("--amp", type=float, default=0.07, help="amplitude contrast")
    ap.add_argument("--voltage", type=float, default=300.0, help="kV")
    ap.add_argument("--oop-step", type=float, default=10.0, help="out-of-plane step (deg)")
    ap.add_argument("--ip-step", type=float, default=7.5, help="in-plane step (deg)")
    ap.add_argument("--symmetry", default="C1")
    ap.add_argument("--defocus-range", type=float, default=1200.0, help="+/- range (A)")
    ap.add_argument("--defocus-step", type=float, default=200.0, help="step (A)")
    ap.add_argument("--whiten-mode", choices=["global", "local"], default="global",
                    help="global = as in cisTEM; local = patch-wise")
    ap.add_argument("--whiten-patch", type=int, default=512, help="local patch size (native px)")
    ap.add_argument("--whiten-stride", type=int, default=256, help="local patch stride (native px)")
    ap.add_argument("--pad-scale", type=float, default=1.0, help="cryojax volume padding")
    ap.add_argument("--batch", type=int, default=200, help="poses per compiled chunk")
    ap.add_argument("--max-orientations", type=int, default=0,
                    help="only search the first N orientations (0 = all)")
    ap.add_argument("--out-prefix", required=True)
    args = ap.parse_args(argv)

    with mrcfile.open(args.mic, permissive=True) as m:
        mic = np.array(m.data.squeeze(), dtype=np.float32)
    with mrcfile.open(args.volume, permissive=True) as m:
        vol = np.array(m.data, dtype=np.float32)
    print(f"mic {mic.shape} @ {args.native_px} A/px, volume {vol.shape}", flush=True)

    maps, info = match_template(
        mic, vol, args.native_px, args.search_px, args.df1, args.df2,
        astig_angle=args.astig_angle, cs=args.cs, amp=args.amp,
        voltage_kv=args.voltage, oop_step=args.oop_step, ip_step=args.ip_step,
        symmetry=args.symmetry, defocus_range=args.defocus_range,
        defocus_step=args.defocus_step, whiten_mode=args.whiten_mode,
        whiten_patch=args.whiten_patch, whiten_stride=args.whiten_stride,
        pad_scale=args.pad_scale, batch=args.batch,
        max_orientations=args.max_orientations)

    Path(args.out_prefix).parent.mkdir(parents=True, exist_ok=True)
    for name, arr in maps.items():
        with mrcfile.new(f"{args.out_prefix}_{name}.mrc", overwrite=True) as mo:
            mo.set_data(arr)
            mo.voxel_size = args.search_px
    np.savez(f"{args.out_prefix}_manifest.npz",
             mic=str(Path(args.mic).resolve()), mic_sha256=file_sha256(args.mic),
             volume=str(Path(args.volume).resolve()),
             volume_sha256=file_sha256(args.volume),
             native_pixel_size=np.float32(args.native_px),
             pixel_size=np.float32(args.search_px),
             search_shape=np.asarray(info["search_shape"], dtype=np.int32),
             whiten_mode=args.whiten_mode, whiten_patch=np.int32(args.whiten_patch),
             whiten_stride=np.int32(args.whiten_stride),
             oop_step=np.float32(args.oop_step), ip_step=np.float32(args.ip_step),
             n_sphere_positions=info["n_sphere_positions"],
             n_psi_rotations=info["n_psi_rotations"], n_poses=info["n_poses"],
             n_defoci=info["n_defoci"], gmean=np.float32(info["gmean"]),
             gstd=np.float32(info["gstd"]),
             **{k: f"{args.out_prefix}_{k}.mrc" for k in maps})
    print(f"saved {args.out_prefix}_*.mrc ({info['seconds']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
