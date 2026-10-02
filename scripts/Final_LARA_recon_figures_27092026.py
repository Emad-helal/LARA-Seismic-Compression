"""
Regenerate reconstruction-quality figures for the COMPLETED Final_LARA_27092026 run.

The published figures show only the first 6 test traces
(`num_samples = min(6, len(original))` in the notebook's plotting helpers).
This script redraws 20 traces, as one `subplots(1, 2)` time|frequency figure
per (model, CR, sample), for CR = 2, 10, 20, 50, 100 and all six models.

Nothing is trained. The three neural models reload their published checkpoints;
the three wavelet baselines are analytic and need no weights.

Fidelity is enforced by two guards:
  1. `class LARA` is sliced verbatim out of the source notebook and must hash to
     the known value, so a drifted notebook aborts instead of quietly drawing
     figures from a different architecture.
  2. Before any figure is written, every neural checkpoint is re-evaluated on the
     full 200-trace evaluation set and the recomputed SNR is compared against the
     published model_comparison_results.csv. A mismatch aborts.

The only directory this script creates is
    <run>\\results\\reconstruction_20samples\\
The pre-existing results tree, including the 6-sample figures, is never written to.
"""

import os
import sys
import json
import math
import time
import hashlib
import warnings

warnings.filterwarnings("ignore")

import logging
logging.getLogger("matplotlib").setLevel(logging.ERROR)
try:                                   # EPS backend chatter, 1200 times over
    import warnings as _w
    _w.filterwarnings("ignore", message=".*PostScript backend.*")
except Exception:
    pass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import torch
import pywt
import h5py

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass


# ====================================================================== paths
RUN_DIR = r"D:\Post_Doctor\Dr. Mostafa\Enhanced_Residual_AutoEncoder\Dr. Omar\Review_27092026\seismic 26 9 2026"
RESULTS_DIR = os.path.join(RUN_DIR, "results")
CKPT_DIR = os.path.join(RUN_DIR, "ckpts")
SOURCE_NOTEBOOK = r"C:\Users\master\Final_HARA\Seismic 26092025\Final_LARA_27092026.ipynb"
PUBLISHED_CSV = os.path.join(RESULTS_DIR, "model_comparison_results.csv")

OUT_DIR = os.path.join(RESULTS_DIR, "reconstruction_20samples")
MANIFEST = os.path.join(OUT_DIR, "figure_manifest.csv")

# ============================================================= study settings
CR_LIST = [2, 10, 20, 50, 100]
N_SHOW = 20

MODEL_SPECS = [
    ("LARA", None, {"variant": "pyramid_funnel32"}),
    ("GeneralizedAutoencoder", None, {}),
    ("AE_PureConcat", None, {}),
    ("Wavelet_DB4", "WaveletDB4", {}),
    ("Wavelet_SYM8", "WaveletSYM8", {}),
    ("Wavelet_COIF3", "WaveletCOIF3", {}),
]

EXPECTED_LARA_SHA = "ea4c8f4b8a2caaa3"   # class LARA text, all three notebook copies
GATE_TOL_DB = 0.05                        # CPU vs CUDA conv differences, in dB

FIG_DPI = 300
FIG_SIZE = (15, 4.2)                      # one row, two panels


# ============================================================ source notebook
def notebook_cells(path):
    with open(path, encoding="utf-8") as fh:
        nb = json.load(fh)
    return {i: "".join(c["source"]) for i, c in enumerate(nb["cells"])
            if c["cell_type"] == "code"}


def lara_class_text(src):
    """Exact span of `class LARA` in the cell that defines it.

    Starts at the class statement itself and stops just before the next
    top-level `class`, or at the end of the cell. This is the same span whose
    sha256 was recorded as ea4c8f4b8a2caaa3 across all three notebook copies.
    """
    i0 = src.index("class LARA(nn.Module)")
    i1 = src.find("\nclass ", i0 + 10)
    return (src[i0:i1] if i1 > 0 else src[i0:]).rstrip()


print("=" * 72)
print("Final_LARA_27092026 - reconstruction figure regeneration")
print("=" * 72)

cells = notebook_cells(SOURCE_NOTEBOOK)

# ---- guard 1: class LARA must be byte-identical to the trained architecture
lara_sha = hashlib.sha256(lara_class_text(cells[6]).encode()).hexdigest()[:16]
if lara_sha != EXPECTED_LARA_SHA:
    raise SystemExit(
        f"ABORT: class LARA hashes to {lara_sha}, expected {EXPECTED_LARA_SHA}.\n"
        f"       {SOURCE_NOTEBOOK} has changed; regenerating figures from a\n"
        f"       different architecture would misrepresent the published run.")
print(f"[guard 1] class LARA sha256 {lara_sha} -> matches the trained architecture")

# ---- build the namespace by executing the notebook's own cells
ns = {"__name__": "lara_recon_figs"}

exec(compile(cells[1], "<nb cell 1: imports>", "exec"), ns)   # imports + seeds

# CONFIG, copied verbatim in value from the notebook's CONFIG cell so that cell 2
# (which creates output directories) never runs.
ns.update({
    "SMOKE": False,
    "CSV_PATH": r"D:\STEAD\merged.csv",
    "H5_PATH": r"D:\STEAD\merged.hdf5",
    "N_TRACES": None,
    "COMPONENT": 2,
    "N_EVAL": 200,
    "MIN_MAGNITUDE": 2.5,
    "MAX_DISTANCE_KM": 60,
    "CRS": [2, 3, 5, 10, 15, 20, 30, 50, 60, 100],
    "BATCH": 16,
    "WORKERS": 0,          # never iterated here; keep the parent process single-process
    "SEED": 42,
    "PIN_MEMORY": False,
    "CKPT_DIR": CKPT_DIR,  # absolute, so best_file() finds the published weights
    "device": torch.device("cpu"),
})
ns["MAIN_MODEL_NAME"] = "LARA"
ns["LARA_VARIANT"] = "pyramid_funnel32"

exec(compile(cells[3], "<nb cell 3: wavelet baselines>", "exec"), ns)   # pure definitions
exec(compile(cells[4], "<nb cell 4: data + split>", "exec"), ns)       # pure definitions
exec(compile(cells[5], "<nb cell 5: baselines + blocks>", "exec"), ns)  # loads the data
exec(compile(cells[6], "<nb cell 6: LARA>", "exec"), ns)               # pure definitions
exec(compile(cells[7], "<nb cell 7: metrics helpers>", "exec"), ns)
exec(compile(cells[9], "<nb cell 9: evaluate_model>", "exec"), ns)

LARA_cls = ns["LARA"]
GeneralizedAutoencoder = ns["GeneralizedAutoencoder"]
AE_PureConcat = ns["AE_PureConcat"]
WaveletDB4 = ns["WaveletDB4"]
WaveletSYM8 = ns["WaveletSYM8"]
WaveletCOIF3 = ns["WaveletCOIF3"]
evaluate_model = ns["evaluate_model"]
calculate_ssim_1d = ns["calculate_ssim_1d"]

waveforms = ns["waveforms"]
test_data = ns["test_data"]          # (200, 1, 1500)
test_data_np = ns["test_data_np"]    # (200, 1500)

assert waveforms.shape[0] == 25091, f"pool changed: {waveforms.shape[0]} != 25091"
assert test_data.shape[0] == 200, f"eval subset changed: {test_data.shape[0]} != 200"
print(f"[data] pool {waveforms.shape[0]:,} traces, evaluation subset {test_data.shape[0]}")
print(f"[data] showing the first {N_SHOW} of those, identical for every model and CR")


def build_model(name, wavelet_cls, kwargs, cr):
    if wavelet_cls is not None:
        return ns[wavelet_cls](cr)
    if name == "LARA":
        return LARA_cls(cr, **kwargs)
    return ns[name](cr, **kwargs)


def ckpt_path(name, cr):
    return os.path.join(CKPT_DIR, f"best_{name}_cr{cr}.pth")


# ================================================= guard 2: faithfulness gate
print("\n" + "=" * 72)
print("GUARD 2 - reloaded checkpoints must reproduce the published metrics")
print("=" * 72)
published = pd.read_csv(PUBLISHED_CSV)
print(f"published rows: {len(published)}  models: {sorted(published['Model'].unique())}\n")

gate_rows, gate_fail = [], []
for name, wcls, kwargs in MODEL_SPECS:
    if wcls is not None:
        continue                     # analytic, nothing to reload
    for cr in CR_LIST:
        model = build_model(name, wcls, kwargs, cr)
        state = torch.load(ckpt_path(name, cr), map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        model.eval()
        m = evaluate_model(model, test_data, cr, name)
        row = published[(published["Model"] == name) & (published["CR"] == cr)]
        pub_snr = float(row["SNR"].iloc[0])
        pub_cr = float(row["Actual_CR"].iloc[0])
        d_snr = abs(m["SNR"] - pub_snr)
        d_cr = abs(m["Actual_CR"] - pub_cr)
        ok = d_snr <= GATE_TOL_DB and d_cr <= 0.5
        gate_rows.append((name, cr, pub_snr, m["SNR"], d_snr, pub_cr, m["Actual_CR"], ok))
        if not ok:
            gate_fail.append((name, cr, pub_snr, m["SNR"], pub_cr, m["Actual_CR"]))
        print(f"  {name:<24} CR={cr:<4} published SNR {pub_snr:7.3f}  "
              f"recomputed {m['SNR']:7.3f}  d={d_snr:6.4f} dB   "
              f"CR {pub_cr:7.3f}/{m['Actual_CR']:7.3f}   {'OK' if ok else 'MISMATCH'}")

if gate_fail:
    detail = "\n".join(f"    {n} CR={c}: published SNR {a:.4f} vs recomputed {b:.4f}; "
                       f"CR {x:.3f} vs {y:.3f}" for n, c, a, b, x, y in gate_fail)
    raise SystemExit(f"ABORT: reloaded checkpoints do not reproduce the published run.\n{detail}")
print(f"\n[guard 2] PASS - worst SNR deviation "
      f"{max(r[4] for r in gate_rows):.5f} dB (tolerance {GATE_TOL_DB})")

# ================================================================= plotting
def per_sample_metrics(orig, rec):
    """Per-trace metrics, same definitions the notebook's evaluate_model uses."""
    m = float(np.mean((orig - rec) ** 2))
    p = float(np.mean(orig ** 2))
    snr = 10 * math.log10(p / (m + 1e-12)) if m > 0 else float("inf")
    c = float(np.corrcoef(orig, rec)[0, 1]) if np.std(orig) > 0 and np.std(rec) > 0 else 0.0
    return snr, m, float(calculate_ssim_1d(orig, rec)), c


def plot_pair(model_name, cr, sample_no, orig, rec):
    """One `subplots(1, 2)` figure: time domain | frequency domain."""
    snr, mse, ssim, corr = per_sample_metrics(orig, rec)
    fig, axes = plt.subplots(1, 2, figsize=FIG_SIZE)

    axes[0].plot(orig, "b-", label="Original", linewidth=1.5, alpha=0.8)
    axes[0].plot(rec, "r-", label="Reconstructed", linewidth=1.5, alpha=0.8)
    axes[0].set_title(f"Time Domain", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Time Samples")
    axes[0].set_ylabel("Amplitude (Normalized)")
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    o_fft = np.abs(np.fft.fft(orig))
    r_fft = np.abs(np.fft.fft(rec))
    freq = np.fft.fftfreq(len(orig))
    pos = freq >= 0
    axes[1].semilogy(freq[pos], o_fft[pos], "b-", label="Original", linewidth=1.5, alpha=0.8)
    axes[1].semilogy(freq[pos], r_fft[pos], "r-", label="Reconstructed", linewidth=1.5, alpha=0.8)
    axes[1].set_title("Frequency Spectrum", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Frequency (Normalized)")
    axes[1].set_ylabel("Magnitude (log scale)")
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(f"{model_name} - CR={cr} - Sample {sample_no} - "
                 f"SNR {snr:.2f} dB | MSE {mse:.6f} | SSIM {ssim:.4f} | Corr {corr:.4f}",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    base = f"{model_name}_CR{cr}_sample{sample_no}_time_frequency"
    png = os.path.join(model_dir, f"{base}.png")
    eps = os.path.join(model_dir, f"{base}.eps")
    fig.savefig(png, dpi=FIG_DPI, bbox_inches="tight", facecolor="white")
    fig.savefig(eps, format="eps", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return os.path.relpath(png, RESULTS_DIR).replace("\\", "/"), snr, mse, ssim, corr


# ================================================================== generate
print("\n" + "=" * 72)
print("GENERATING FIGURES")
print("=" * 72)
os.makedirs(OUT_DIR, exist_ok=True)

manifest, recon_cache = [], {}
t0 = time.time()
done = 0

for model_name, wcls, kwargs in MODEL_SPECS:
    model_dir = os.path.join(OUT_DIR, model_name)
    os.makedirs(model_dir, exist_ok=True)
    print(f"\n--- {model_name} ---")

    for cr in CR_LIST:
        model = build_model(model_name, wcls, kwargs, cr)
        if wcls is None:
            state = torch.load(ckpt_path(model_name, cr), map_location="cpu", weights_only=True)
            model.load_state_dict(state)
            model.eval()
            with torch.no_grad():
                x = torch.tensor(test_data[:N_SHOW], dtype=torch.float32)
                rec = model(x).cpu().numpy()
            if rec.ndim == 3:
                rec = rec[:, 0, :]
        else:
            rec = np.asarray([model.compress_decompress(s) for s in test_data_np[:N_SHOW]])

        recon_cache[(model_name, cr)] = rec
        for i in range(N_SHOW):
            rel, snr, mse, ssim, corr = plot_pair(
                model_name, cr, i + 1, test_data_np[i], rec[i])
            manifest.append({
                "figure": rel, "Model": model_name, "CR": cr, "Sample": i + 1,
                "SNR_dB": round(snr, 4), "MSE": round(mse, 8),
                "SSIM": round(ssim, 6), "Correlation": round(corr, 6),
            })
            done += 1
        print(f"    CR={cr:<4} {N_SHOW} figures")

man_df = pd.DataFrame(manifest)
man_df.to_csv(MANIFEST, index=False)

elapsed = time.time() - t0
print("\n" + "=" * 72)
print(f"wrote {done} figures ({done * 2} files counting EPS) in {elapsed / 60:.1f} min")
print(f"manifest: {MANIFEST}  ({len(man_df)} rows)")
print(f"output  : {OUT_DIR}")
print("=" * 72)
