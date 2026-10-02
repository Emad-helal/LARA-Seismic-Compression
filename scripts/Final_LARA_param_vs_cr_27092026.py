"""
Parameter Count vs Compression Ratio for the three neural models of the
completed Final_LARA_27092026 run.

Reads model_comparison_results.csv, keeps LARA / GeneralizedAutoencoder /
AE_PureConcat, and writes two figures - linear y and log y - because the
parameter counts span 0.38M to 19.13M (a 50x range) and on a linear axis
everything from CR >= 30 collapses into the bottom 6% of the plot, which is
exactly where all four curve crossovers happen.

The three wavelet baselines are excluded: they are recorded with
Parameter_Count == 0 (they have no learnable parameters), so including them
would flatten three lines onto the x-axis and waste the legend.

Colours reuse slots 0/1/2 of the same 6-model Set3 palette that
plot_parameter_comparison used, so the model-to-colour mapping matches the
existing model_analysis\\parameter_count_comparison.png.

Pure CSV read and matplotlib: no models, no checkpoints, no data loading.
"""

import os
import sys
import json
import warnings

warnings.filterwarnings("ignore")
import logging
logging.getLogger("matplotlib").setLevel(logging.ERROR)
try:
    import warnings as _w
    _w.filterwarnings("ignore", message=".*PostScript backend.*")
except Exception:
    pass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass


RUN_DIR = r"D:\Post_Doctor\Dr. Mostafa\Enhanced_Residual_AutoEncoder\Dr. Omar\Review_27092026\seismic 26 9 2026"
RESULTS_DIR = os.path.join(RUN_DIR, "results")
SOURCE_CSV = os.path.join(RESULTS_DIR, "model_comparison_results.csv")
ANALYSIS_DIR = os.path.join(RESULTS_DIR, "model_analysis")

MODELS = ["LARA", "GeneralizedAutoencoder", "AE_PureConcat"]
EXPECTED_CRS = [2, 3, 5, 10, 15, 20, 30, 50, 60, 100]

# Set3 slots 0, 1, 2 of the 6-model palette used by the published figure:
# LARA teal, GeneralizedAutoencoder lavender, AE_PureConcat steel blue.
FULL_PALETTE_SIZE = 6
COLORS = {m: plt.cm.Set3(i / (FULL_PALETTE_SIZE - 1)) for i, m in enumerate(MODELS)}
MARKERS = {m: mkr for m, mkr in zip(MODELS, ["o", "s", "^"])}

FIG_SIZE = (14, 8)


# ------------------------------------------------------------------ load
print("=" * 72)
print("Parameter Count vs Compression Ratio - neural models")
print("=" * 72)

df = pd.read_csv(SOURCE_CSV)
print(f"source: {SOURCE_CSV}")
print(f"loaded {len(df)} rows, models {sorted(df['Model'].unique())}")

sub = df[df["Model"].isin(MODELS)].copy()

# ---- assertions: fail loudly rather than draw a misleading figure ----
if len(sub) != 30:
    raise SystemExit(f"ABORT: expected 30 rows for the 3 neural models, got {len(sub)}")

crs = sorted(sub["CR"].unique().tolist())
if crs != EXPECTED_CRS:
    raise SystemExit(f"ABORT: CR set changed: {crs} != {EXPECTED_CRS}")

counts = sub["Parameter_Count"].to_numpy(dtype=float)
if not np.all(counts > 0):
    zero = sub.loc[sub["Parameter_Count"] <= 0, ["Model", "CR"]].to_dict("records")
    raise SystemExit(f"ABORT: non-positive parameter counts reached the neural set: {zero}")
if not np.all(counts == np.round(counts)):
    raise SystemExit("ABORT: Parameter_Count is not integral")

piv = (sub.pivot_table(index="CR", columns="Model", values="Parameter_Count")[MODELS] / 1e6)
if piv.shape != (10, 3) or piv.isna().any().any():
    raise SystemExit(f"ABORT: pivot incomplete, shape {piv.shape}, NaNs {int(piv.isna().sum().sum())}")

print(f"\nverified: 30 rows, CRs {crs}, all counts positive integers, pivot {piv.shape} complete")
print("\nParameter Count (millions):")
print(piv.round(4).to_string())


# ------------------------------------------------------------------ plot
def make_figure(log_y):
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    for m in MODELS:
        ax.semilogx(piv.index, piv[m], marker=MARKERS[m], linestyle="-",
                    color=COLORS[m], label=m, markersize=8, linewidth=2,
                    markerfacecolor=COLORS[m], markeredgecolor="black")

    if log_y:
        ax.set_yscale("log")
    else:
        ax.set_yscale("linear")

    # every CR labelled, so no point is hidden between decades
    ax.set_xticks(piv.index)
    ax.set_xticklabels([str(c) for c in piv.index])
    ax.minorticks_off()

    ax.set_xlabel("Compression Ratio", fontsize=14)
    ax.set_ylabel("Parameter Count ($\\times 10^{6}$)", fontsize=14)
    ax.set_title("Parameter Count vs Compression Ratio - Neural Models",
                 fontsize=16, fontweight="bold")
    ax.grid(True, alpha=0.3, which="major")
    ax.legend(fontsize=12, loc="upper right")
    fig.tight_layout()

    tag = "log" if log_y else "linear"
    base = os.path.join(ANALYSIS_DIR, f"parameter_count_vs_cr_neural3_{tag}")
    png, eps = f"{base}.png", f"{base}.eps"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(eps, format="eps", bbox_inches="tight")
    plt.close(fig)
    return png, eps


def make_bar_figure():
    """Grouped bar chart, single panel, linear y, equally spaced CR groups.

    Linear y is deliberate: a bar's length encodes magnitude measured from zero,
    and a log axis has no zero, so on a log scale the bar bases would be
    arbitrary and the lengths would stop being proportional to the values.
    The cost is that the CR >= 30 groups sit low against the 0-20M axis; the
    log-scaled line figure covers that regime honestly.
    """
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    x = np.arange(len(piv.index))
    w = 0.26
    for i, m in enumerate(MODELS):
        ax.bar(x + (i - 1) * w, piv[m].to_numpy(), width=w, color=COLORS[m],
               label=m, edgecolor="black", linewidth=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels([str(c) for c in piv.index])
    ax.set_xlabel("Compression Ratio", fontsize=14)
    ax.set_ylabel("Parameter Count ($\\times 10^{6}$)", fontsize=14)
    ax.set_title("Parameter Count by Compression Ratio - Neural Models",
                 fontsize=16, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(fontsize=12, loc="upper right")
    fig.tight_layout()

    base = os.path.join(ANALYSIS_DIR, "parameter_count_vs_cr_neural3_bar")
    png, eps = f"{base}.png", f"{base}.eps"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(eps, format="eps", bbox_inches="tight")
    plt.close(fig)
    return png, eps


os.makedirs(ANALYSIS_DIR, exist_ok=True)
print()
written = [make_figure(log_y) for log_y in (False, True)]
written.append(make_bar_figure())
for png, eps in written:
    for p in (png, eps):
        size = os.path.getsize(p)
        if size == 0:
            raise SystemExit(f"ABORT: {p} is zero bytes")
        print(f"  {size:>9} bytes  {os.path.basename(p)}")

# ---- independent re-read: confirm the files sit on the numbers in the CSV
print("\nindependent re-read of the saved CSV:")
re_df = pd.read_csv(SOURCE_CSV)
re_piv = (re_df[re_df["Model"].isin(MODELS)]
          .pivot_table(index="CR", columns="Model", values="Parameter_Count")[MODELS] / 1e6)
mismatch = int((re_piv.round(6) != piv.round(6)).sum().sum())
if mismatch:
    raise SystemExit(f"ABORT: re-read disagrees with the plotted table in {mismatch} cells")
print(f"  {mismatch} mismatched cells out of {piv.size} - plotted values match the CSV")

print("\ndone.")
