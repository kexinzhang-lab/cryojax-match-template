# cryojax-match-template

A JAX port of cisTEM's `match_template` (full-micrograph 2D template matching),
using [cryojax](https://github.com/michael-0brien/cryojax) as the projector.

The preprocessing, template normalization, correlation and MIP / scaled-MIP
statistics follow cisTEM's `match_template`, so outputs can be compared directly.
Search pixel size, angular sampling and defocus range are all plain arguments,
and everything is in Python, which makes it easy to modify.

## Install

```bash
# in an environment with a working GPU build of jax
pip install cryojax
pip install -e .
```

## Usage

```bash
cryojax-match-template \
    --mic micrograph.mrc --native-px 1.06 \
    --volume template.mrc --search-px 1.06 \
    --df1 4305 --df2 4066 --astig-angle 16.8 \
    --oop-step 10 --ip-step 7.5 \
    --defocus-range 1200 --defocus-step 200 \
    --out-prefix results/mic147
```

The template volume must already be at `--search-px`. The micrograph is
Fourier-cropped from `--native-px` to `--search-px`.

Use `--max-orientations 100` for a quick test.

Outputs (`<prefix>_*.mrc`, all at the search pixel size):

| file | content |
|---|---|
| `mip` | max over orientations/defoci, z-scored by global CCC mean/std |
| `scaled_mip` | `(mip - avg) / std` per pixel |
| `avg`, `std` | per-pixel mean/std of the CCC over the whole search |
| `phi`, `theta`, `psi` | best orientation (cisTEM ZYZ, degrees) |
| `defocus` | best defocus offset from df1/df2 (A) |

plus `<prefix>_manifest.npz` with the run parameters.

From Python:

```python
from cryojax_match_template import match_template
maps, info = match_template(mic, volume, native_px=1.06, search_px=1.06,
                            df1=4305, df2=4066, astig_angle=16.8)
```

## Method

Image:
1. replace 5σ outliers with the mean
2. whiten with 1/sqrt(radial PSD) (normalized to max 1), zero DC, unit mean-square
3. Fourier-crop to the search pixel size
4. swap quadrants, zero DC, normalize by sqrt(sum of squares / N)

Template, for each orientation and defocus:
1. project with cryojax, CTF applied
2. multiply by the same whitening curve (mapped by physical frequency)
3. subtract the edge mean, then scale to unit variance over the padded image box
4. pad to image size, centered (as `ClipIntoLargerRealSpace2D`)

Correlation is `irfft(conj(T) * I)`. Max, argmax pose/defocus, sum and sum of
squares are accumulated on the GPU; the full search for one micrograph runs on
one GPU without writing intermediate results.

## Differences from cisTEM

- projector is cryojax's Fourier-slice `FourierVoxelGridVolume`. Its Euler
  convention has phi and psi swapped relative to cisTEM, and its astigmatism
  angle is 90° minus cisTEM's. Both are converted internally; all inputs and
  outputs use cisTEM's conventions (defocus angle as in CTFFIND/cisTEM).
- `--whiten-mode local` (optional, not in cisTEM): patch-wise whitening of the
  micrograph, so that regions with different local spectra (thick ice, vacuoles)
  don't dominate the background. The template still uses the global curve.
- no pixel-size search
- one micrograph per GPU; to process many micrographs, run one job per GPU

## Validation

Compared against cisTEM `match_template` on an in-situ 60S ribosome micrograph
(EMPIAR-10998, 1.06 A/px, 10°/7.5°, no defocus search, same template), both at
1.06 A/px:

| | cisTEM | this repo |
|---|---|---|
| scaled MIP mean / std | 4.012 / 0.292 | 4.012 / 0.292 |
| pixel-wise correlation, MIP / scaled MIP | | 0.992 / 0.990 |
| cisTEM peaks > 6.5σ also > 6.5σ here (±2 px) | 21 | 20 |
| pose difference at those peaks | | median 0.0°, all < 10° |

Search time on one RTX 3090: about 2 min for this micrograph
(4092×5760, 20,482 orientations, one defocus).

To reproduce with your own cisTEM run:

```bash
python scripts/debug_single_pose.py ...   # top cisTEM peaks, one template each
python scripts/compare_to_cistem.py --ours results/mic147 \
    --cistem '/path/TemplateMatching/<name>_{}_<suffix>.mrc'
```
