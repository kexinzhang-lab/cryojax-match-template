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
  convention has phi and psi swapped relative to cisTEM; this is handled
  internally, and inputs/outputs use cisTEM's convention.
- `--whiten-mode local` (optional, not in cisTEM): patch-wise whitening of the
  micrograph, so that regions with different local spectra (thick ice, vacuoles)
  don't dominate the background. The template still uses the global curve.
- no pixel-size search
- one micrograph per GPU; to process many micrographs, run one job per GPU

## Validation

Tested against a cisTEM `match_template` run on an in-situ 60S ribosome
micrograph (1.06 A/px, 10°/7.5°, defocus ±1200/200 A):

- scaled-MIP background mean/std: 4.53/0.27 for both
- 12/13 peaks above 7σ found at the same pixel, median pose difference 0.0°
- defocus agrees within one step

Runtime for that search was about 17 min per micrograph on one GPU.
