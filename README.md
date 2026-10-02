# LARA — Seismic Waveform Compression

Reference implementation for **LARA**, the *Light Autoencoder for Reconstruction
and Analysis*, a fixed-length codec for seismic waveforms submitted to
**Scientific Reports**.

> **This release supersedes the earlier `HARA` code base.** The repository was
> renamed `HARA-Seismic-Compression` → `LARA-Seismic-Compression`; GitHub keeps a
> permanent redirect from the old name. The archived **v1.0.0** at
> <https://doi.org/10.5281/zenodo.20172322> is retained as the citable version of
> record for that earlier work. Several of its settings were superseded — see
> [Differences from the archived version](#differences-from-the-archived-version).

---

## Result in one table

Reconstruction SNR in dB on the held-out STEAD evaluation set (mean of per-trace
values; standard deviation across traces in parentheses). Full table, including
CC, SSIM and PSNR, is `results/model_comparison_results.csv`.

| Model | CR=2 | CR=5 | CR=20 | CR=100 |
|---|---|---|---|---|
| **LARA** | 39.66 (5.39) | 31.17 (7.04) | 19.28 (3.40) | 16.16 (2.49) |
| Generalized Autoencoder | 38.15 (5.29) | 29.47 (5.92) | 18.94 (3.32) | 15.87 (2.40) |
| AE_PureConcat | 34.99 (3.91) | 22.65 (5.07) | 16.95 (2.66) | 15.71 (2.30) |
| Wavelet db4 | 60.02 (14.09) | 35.97 (7.07) | 21.92 (3.80) | 3.05 (0.75) |

Three findings worth stating plainly, because each contradicts something the
earlier version of this work reported:

1. **There is no optimal intermediate ratio.** SNR decreases monotonically with
   the compression ratio across all ten ratios tested. The earlier version
   reported a peak at CR=20; that peak is not present on this training pool.
2. **LARA is not the best method at every ratio.** The wavelet compressors are
   more accurate up to a ratio of about 30 and are free of trained parameters.
   Above CR≈30 they fail (correlation below 0.07 at CR=50), and LARA is the only
   method tested that still preserves waveform shape.
3. **Parameter count and accuracy are both regime-dependent.** LARA is the
   smallest learned model below CR=20, and the most accurate learned model at
   every ratio.

---

## Repository layout

```
LARA-Seismic-Compression/
├── notebooks/
│   ├── Final_LARA_27092026.ipynb        # main study: LARA, 2 baselines, 3 wavelet codecs
│   └── Final_Ablation_27092026.ipynb     # 5-arm component ablation
├── scripts/
│   ├── Final_LARA_pipeline_27092026.py       # architecture figure (Fig. 1)
│   ├── Final_LARA_param_vs_cr_27092026.py    # parameter-count figure (Fig. 6)
│   ├── Final_LARA_recon_figures_27092026.py  # reconstruction figures (Figs. 3-5)
│   ├── Final_LARA_bps_analysis_27092026.py   # bit-rate analysis (§2.9, Table 4)
│   ├── arithcoder.py                         # range coder used by the above
│   ├── Final_LARA_compute_bench_27092026.py  # FLOPs / CPU / GPU cost (§2.10)
│   ├── extract_subnormal_fraction.py         # §2.10 subnormal claim, from checkpoints only
│   ├── build_loss_ablation_LARA_27092026.py  # 7-objective loss ablation (§2.8, Table 3)
│   ├── Final_Ablation_27092026_build.py      # assembles the component-ablation notebook
│   ├── Final_Ablation_27092026_runcells.py   # notebook cell splitter used by the above
│   ├── build_bps_docx.py, build_cost_docx.py, build_cost_gpu_docx.py
│   │                                         # builders for the technical notes
│   ├── build_reviewer_response_docx.py       # letter responding to the reviewers
│   ├── fix_bench_csv_units_27092026.py       # unit harmonisation for the cost CSVs
│   ├── treemanifest.py                       # directory inventory
│   └── audit_numbers.py                      # re-derives every claim in the paper
├── results/                                  # the CSVs the paper is audited against
├── figures/                                  # corrected and newly added figures
├── paper/
│   ├── main.tex, Myreferences.bib, wlscirep.cls
├── requirements.txt
└── LICENSE                                  # MIT
```

Notebook outputs have been stripped so the repository stays small; every cell
re-executes to regenerate them. Checkpoints are **not** redistributed — they are
several hundred megabytes — but `extract_subnormal_fraction.py` will read them if
you place them under `seismic 26 9 2026/ckpts/`.

---

## Setup

```bash
git clone https://github.com/Emad-helal/LARA-Seismic-Compression.git
cd LARA-Seismic-Compression
pip install -r requirements.txt
```

### Dataset

The **STanford Earthquake Dataset** (STEAD), filtered to local earthquakes within
60 km and magnitude greater than 2.5. Download from
<https://github.com/smousavi05/STEAD>.

`merged.csv` and `merged.hdf5` are not redistributable, so they are not included.
Place them on the default path or override it:

```python
csv_file  = r"D:\STEAD\merged.csv"
file_name = r"D:\STEAD\merged.hdf5"
```

---

## Reproducing the main study

```python
import os
os.environ["FINAL_LARA_SMOKE"] = "0"   # "1" for a fast end-to-end smoke test
```

then run `notebooks/Final_LARA_27092026.ipynb` top to bottom.

| | |
|---|---|
| Magnitude filter | > 2.5 |
| Distance filter | ≤ 60 km |
| Component | vertical only |
| Traces after filtering | 25,091 |
| Split | 80 / 10 / 10 → 20,073 / 2,509 / 2,509 |
| Compression ratios | 2, 3, 5, 10, 15, 20, 30, 50, 60, 100 |
| Epochs | 120, uniform across models and ratios |
| Batch size | 64 |
| Seed | 42 (both split stages) |

### The architecture

LARA is a `pyramid_funnel32` residual autoencoder with a **flat** bottleneck.
The latent is a vector in **R^(1500/CR)** — it is not a spatial map, so no
downsampling factor applies to it.

| | |
|---|---|
| Stem | 32 channels |
| Residual blocks | 32 → 64 → 128 → (256 when CR > 30) |
| Attention | Channel-wise, on each residual block |
| Bottleneck | Single projection, then **filled** to the target latent length |
| Decoder | Six-level channel pyramid, **not** a mirror of the encoder |

One model is trained per compression ratio; the widths do not change with the
ratio, only the latent length does.

### Baselines

`GeneralizedAutoencoder`, `AE_PureConcat`, and the wavelet codecs **db4**,
**sym8**, **coif3**.

---

## Verifying the published numbers

This is the part we would ask a reviewer to run first.

```bash
python scripts/audit_numbers.py
```

It re-derives **243 quantitative claims** — every value in the results tables,
the prose, and the analysis sections — from the CSVs in `results/`, and exits
non-zero if any of them disagrees with `paper/main.tex`. On this release:

```
CHECKS RUN: 243    FAILURES: 0
```

The paper quotes this count, and the script asserts that the quoted number still
matches its own run, so the two cannot drift apart silently.

The checks this enforces include:

- **Exact compression ratio.** Every trained model's latent width is *measured*,
  not assumed. A run aborts if any model disagrees with its target ratio.
- **Architecture fingerprint.** The analysis scripts embed the `LARA` class and
  verify its SHA-256 digest against the trained architecture, so a figure cannot
  be produced from a model that has drifted from the one that was trained.
- **Checkpoint re-verification.** The bit-rate and loss-ablation scripts reload
  the published checkpoints, recompute the reported reconstruction quality, and
  compare it against the published table before drawing anything.
- **Every cell of the results tables, not just spot checks.** All 96 cells of
  Table 1, including the standard-deviation rows, are re-derived from
  `model_comparison_results.csv`.
- **The subnormal-weight finding, recomputed.** §2.10 reports that 23.1–52.9% of
  LARA's stored parameters at CR ≥ 50 are subnormal, and exactly 0.00% for both
  baselines. That quantity depends only on the tensors, so
  `extract_subnormal_fraction.py` reproduces it straight from the checkpoints:

  ```bash
  python scripts/extract_subnormal_fraction.py
  ```

---

## The loss-function ablation

`scripts/build_loss_ablation_LARA_27092026.py` trains LARA from scratch under
seven objectives — MSE, STFT, STFT+arrival, STFT+phase, STFT+wavelet,
STFT+phase+arrival, NCC+arrival — over the full ratio grid, with deterministic
cuDNN kernels.

```bash
python scripts/build_loss_ablation_LARA_27092026.py --smoke     # quick check
python scripts/build_loss_ablation_LARA_27092026.py --cr 5,10   # real runs
```

`--data-csv` / `--data-h5` override the dataset location. This script does not
load the published checkpoints: each arm is trained independently, so the
comparison is between objectives and not between initialisations.

---

## Known limitations

These are properties of the study, stated so they are not mistaken for results:

- **Single run per configuration.** Every reported error bar is the dispersion
  **across evaluation traces**, not across training runs. Several margins,
  including the 0.22–0.52 dB by which LARA leads the better baseline above CR=10,
  are of the same order as that dispersion and are not claimed to be individually
  significant. Separating the two requires repeated training.
- **Absolute amplitude is not preserved.** Traces are rescaled to `[0, 1]` by
  their own min/max before encoding. The codec preserves waveform *shape*.
- **Distribution shift is untested on the main models.** The zero-shot
  (TXED, INSTANCE) and distributed acoustic sensing (DAS) sections were produced
  with the earlier heavy-capacity configuration, not this one.
- **Most of the bit-rate gain is quantization, not entropy coding.** Quantizer
  depth gives 4.14–9.04× at ≤0.41 dB; the arithmetic coder's own contribution
  over fixed-width ranges from −6% to +13%, and is *worse* than fixed-width at
  CR 50, 60 and 100.
- **Subnormal weights at high ratios.** 23.1–52.9% of LARA's stored float32
  parameters lie in the subnormal range at CR ≥ 50. On a CPU without
  flush-to-zero this is severe (CR=50 encode: 2.78 ms flushed vs 1328.60 ms as
  deployed). It affects the trained weights, not the architecture.

---

## Differences from the archived version

The archived v1.0.0 README describes a different experimental setup. None of the
following are the settings used here:

| | Archived v1.0.0 | This release |
|---|---|---|
| Magnitude threshold | > 3 | **> 2.5** |
| Split | 90 / 10 | **80 / 10 / 10** |
| Epoch budget | ratio-dependent | **120, uniform** |
| Peak SNR | 35.41 dB at CR=20 | **monotonic decrease; no peak** |
| Capacity | heavy | **light** (`pyramid_funnel32`) |

The v1.0.0 numbers are not reproduced here and are not cited by the paper. The
archive is kept only because it is the version of record for the earlier work,
and because the external-dataset and DAS results quoted in that earlier work
were obtained with it.

---

## Citation

```bibtex
@article{helal2026lara,
  title   = {Seismic Waveform Compression Using a Light Attention-Enhanced
             Residual Autoencoder},
  author  = {Helal, Emad B. and Hafez, Ali G. and Khan, Rizwan and
             Ibrahim, Mostafa M.},
  journal = {Scientific Reports},
  year    = {2026},
  note    = {Under review}
}
```

## License

MIT — see [`LICENSE`](LICENSE).