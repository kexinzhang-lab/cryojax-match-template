# Comparison with cisTEM match_template: EMPIAR-10998 micrograph 147

Date: 2026-10-05

## Setup

| | |
|---|---|
| micrograph | EMPIAR-10998 `141_Mar12_11.57.31_147_0.mrc`, 5760 × 4092, 1.06 Å/px |
| template | `parsed_6Q8Y_whole_LSU_match3_sim_60.mrc` (yeast 60S), 384³, 1.06 Å/px |
| CTF | df1 6307.9 Å, df2 5296.5 Å, angle −66.06°, 300 kV, Cs 2.7 mm, amp. contrast 0.07 |
| search pixel size | 1.06 Å (no binning) |
| angular grid | C1, out-of-plane 10°, in-plane 7.5°: 418 sphere positions × 49 psi = **20,482 orientations** |
| defocus search | none (single defocus) |
| cisTEM-only settings | high-res limit 3.0 Å, padding 1.0, mask radius 0 (max), 4 threads |

This is the same search as cisTEM job 40 in the original project (`..._29_3.mrc` outputs).

## Hardware and software

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 3090 (24 GB), driver 535.183.01, CUDA 12.2 |
| CPU | Intel i9-12900KF, 24 threads |
| cisTEM | `match_template` 2.0.0-alpha-249-d9bf2dd (je_dev3 build, 2024-05-20), GPU |
| cryojax-match-template | commit `777a32e`; python 3.11.15, jax/jaxlib 0.9.2, cryojax 0.5.6.dev6, `--batch 200`, `--pad-scale 1.0` |

For reference, an older cisTEM build was also timed: `cisTEM_downstream_bah` (compiled 2021-08-10), GPU.

## Timing

Wall-clock for the full run (read inputs, search, write outputs), same GPU, nothing else running on it.

| | run 1 | run 2 |
|---|---|---|
| cisTEM je_dev3 | 68.9 s | 73.7 s |
| cisTEM 2021 build | 88.8 s | |
| cryojax-match-template | 115 s | 123 s |

For cryojax, about 10 s of each run is setup (whitening, JAX compilation); the search itself runs
at ~5.1 ms per orientation (~200 orientations/s) on the 4092 × 5760 image. At a 1.5 Å search pixel size
(2898 × 4080 image) it is ~2.5 ms per orientation, 58 s total.

Overall cryojax is ~1.7× slower than the je_dev3 build for this search.

## Accuracy

### Agreement with cisTEM

`scripts/compare_to_cistem.py`, 200 px edge excluded, peaks matched within 2 px.

| | per-pixel r (MIP / scaled MIP) | scaled MIP mean / std | peaks > 6.5σ also > 6.5σ | pose at those peaks |
|---|---|---|---|---|
| cisTEM je_dev3 rerun vs. original job 40 | 1.000 / 1.000 | 4.012 / 0.292 vs 4.012 / 0.292 | 21 / 21 | identical |
| cisTEM 2021 build vs. job 40 | 0.999 / 0.999 | 4.012 / 0.292 | 21 / 21 | identical |
| **cryojax vs. cisTEM je_dev3** | **0.992 / 0.990** | 4.012 / 0.292 vs 4.012 / 0.292 | **20 / 21** | median 0.0°, all < 10° |

### Against a reference particle list

Reference: the 85 peaks (> 7.85σ) of a fine cisTEM search on the same micrograph
(job 36: 2.5° / 1.5°, defocus ±1200 Å in 200 Å steps). 82 are away from the edge and are scored.
`scripts/score_vs_reference.py`, match radius 10 px.

| | recall in top 85 / 200 / 500 / 1000 peaks | peaks > 7.85σ | peaks > 7.0σ (precision) | scaled MIP at reference particles (median) |
|---|---|---|---|---|
| cisTEM je_dev3 | 0.02 / 0.02 / 0.05 / 0.07 | 0 | 4 (0.25) | 4.10 |
| cryojax | 0.02 / 0.02 / 0.05 / 0.07 | 0 | 3 (0.00) | 4.08 |

Both implementations behave the same, and neither detects these particles: at 1.06 Å, a 10° / 7.5°
grid without defocus search is too coarse, and the scaled MIP at the true particle positions is at
background level (~4.1). This search setting is therefore useful for checking that the two
implementations agree, but not for comparing detection accuracy. That needs a finer angular grid.

## Reproduce

```bash
MIC=141_Mar12_11.57.31_147_0.mrc
VOL=parsed_6Q8Y_whole_LSU_match3_sim_60.mrc

# cryojax
cryojax-match-template --mic $MIC --volume $VOL --native-px 1.06 --search-px 1.06 \
    --df1 6307.929199 --df2 5296.518555 --astig-angle -66.057884 \
    --oop-step 10 --ip-step 7.5 --defocus-range 0 --out-prefix out/cryojax

# cisTEM
scripts/run_cistem_match_template.sh /path/to/match_template $MIC $VOL out/cistem \
    1.06 6307.929199 5296.518555 -66.057884 3.0 10 7.5 0 0

# compare
python scripts/compare_to_cistem.py --ours out/cryojax --cistem 'out/cistem_{}.mrc'
python scripts/score_vs_reference.py --ref-db project.db --ref-id 84 --pixel-size 1.06 \
    "cisTEM=out/cistem_{}.mrc" "cryojax=out/cryojax_{}.mrc"
```
