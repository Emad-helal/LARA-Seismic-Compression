"""
End-to-end BPS analysis for the neural models of the completed
Final_LARA_27092026 run.

The manuscript reports a compression ratio defined as 1500 / latent_dim, with
no quantization and no entropy coding. That makes the number a float32 latent
storage ratio: BPS = 32 / CR exactly, with zero coding gain. This script
measures what the latent actually costs once it is quantized and entropy coded,
which is the number a reviewer asking for "effective bits-per-sample" wants.

For every (model, CR):
  1. rebuild the published split and reload the published checkpoint,
  2. confirm the recomputed 200-trace SNR still matches
     model_comparison_results.csv (the faithfulness gate),
  3. extract the latent code, exactly 1500 // CR values,
  4. calibrate a per-dimension quantizer range on the TRAIN split only,
  5. for b in BITS: quantize, entropy code, decode, reconstruct, measure quality,
  6. account for every bit, including the model weights and the range side info.

The entropy coder is a real adaptive arithmetic coder with a matching decoder.
Every coded trace is decoded back and compared against the quantized symbols,
so a rate is only ever reported for data the decoder can actually recover.

Output: results/bps_analysis/
"""

import os
import sys
import json
import time
import math
import hashlib
import argparse
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
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import torch
import h5py
import pywt

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
OUT_DIR = os.path.join(RESULTS_DIR, "bps_analysis")

MODELS = ["LARA", "GeneralizedAutoencoder", "AE_PureConcat"]
EXPECTED_CRS = [2, 3, 5, 10, 15, 20, 30, 50, 60, 100]
EXPECTED_LARA_SHA = "ea4c8f4b8a2caaa3"
GATE_TOL_DB = 0.05

N_SAMPLES = 1500
SOURCE_BITS_PER_SAMPLE = 32          # merged.hdf5 stores float32
CALIB_N = 2000                      # training traces used to set quantizer ranges
BITS = [8, 6, 5, 4, 3, 2]           # quantizer precision sweep


# ============================================== adaptive arithmetic coder
# Imported, not inlined: the coder is unit-tested separately for round-trip
# correctness and for closeness to the zeroth-order entropy bound. A BPS
# number is only meaningful if a decoder can actually recover the trace.
from arithcoder import measure


# ============================================================ source notebook
def notebook_cells(path):
    with open(path, encoding="utf-8") as fh:
        return {i: "".join(c["source"]) for i, c in enumerate(json.load(fh)["cells"])
                if c["cell_type"] == "code"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="one model, two CRs, three bit depths - a fast end-to-end check")
    ap.add_argument("--rate-traces", type=int, default=200,
                    help="test traces used for the coded-rate measurement")
    args = ap.parse_args()

    print("=" * 74)
    print("Final_LARA_27092026 - end-to-end BPS analysis (neural models)")
    print("=" * 74)

    cells = notebook_cells(SOURCE_NOTEBOOK)
    lara_text = cells[6]
    i0 = lara_text.index("class LARA(nn.Module)")
    i1 = lara_text.find("\nclass ", i0 + 10)
    lara_sha = hashlib.sha256(
        ((lara_text[i0:i1] if i1 > 0 else lara_text[i0:]).rstrip()).encode()).hexdigest()[:16]
    if lara_sha != EXPECTED_LARA_SHA:
        raise SystemExit(f"ABORT: class LARA hashes to {lara_sha}, expected {EXPECTED_LARA_SHA}")
    print(f"[guard 1] class LARA sha256 {lara_sha} -> matches the trained architecture")

    ns = {"__name__": "lara_bps"}
    exec(compile(cells[1], "<nb cell 1: imports>", "exec"), ns)
    ns.update({
        "SMOKE": False, "CSV_PATH": r"D:\STEAD\merged.csv", "H5_PATH": r"D:\STEAD\merged.hdf5",
        "N_TRACES": None, "COMPONENT": 2, "N_EVAL": 200, "MIN_MAGNITUDE": 2.5,
        "MAX_DISTANCE_KM": 60, "CRS": EXPECTED_CRS, "BATCH": 16, "WORKERS": 0,
        "SEED": 42, "PIN_MEMORY": False, "CKPT_DIR": CKPT_DIR,
        "device": torch.device("cpu"),
    })
    ns["MAIN_MODEL_NAME"] = "LARA"
    ns["LARA_VARIANT"] = "pyramid_funnel32"
    for ci, tag in ((3, "wavelet baselines"), (4, "data + split"), (5, "baselines + blocks"),
                    (6, "LARA"), (7, "metric helpers"), (9, "evaluate_model")):
        exec(compile(cells[ci], f"<nb cell {ci}: {tag}>", "exec"), ns)

    LARA_cls = ns["LARA"]
    GeneralizedAutoencoder = ns["GeneralizedAutoencoder"]
    AE_PureConcat = ns["AE_PureConcat"]
    evaluate_model = ns["evaluate_model"]
    calculate_ssim_1d = ns["calculate_ssim_1d"]
    test_data = ns["test_data"]
    test_data_np = ns["test_data_np"]
    train_dataset = ns["train_dataset"]
    waveforms = ns["waveforms"]

    assert waveforms.shape[0] == 25091, f"pool changed: {waveforms.shape[0]}"
    assert test_data.shape[0] == 200, f"eval subset changed: {test_data.shape[0]}"

    calib_x = torch.stack([train_dataset[i] for i in range(min(CALIB_N, len(train_dataset)))])
    print(f"[data] pool {waveforms.shape[0]:,}, test {test_data.shape[0]}, "
          f"calibration {calib_x.shape[0]} training traces (train split only)")

    # ---------------------------------------------------------- model plumbing
    def build(name, cr):
        if name == "LARA":
            m = LARA_cls(cr, variant="pyramid_funnel32")
            assert getattr(m, "variant", None) == "pyramid_funnel32"
            return m
        if name == "GeneralizedAutoencoder":
            return GeneralizedAutoencoder(cr)
        return AE_PureConcat(cr)

    def encode_latent(model, name, x):
        with torch.no_grad():
            if name in ("LARA", "GeneralizedAutoencoder"):
                return model.encoder(x)
            return torch.cat([b(x) for b in model.branches], dim=1)

    def decode_latent(model, name, z):
        with torch.no_grad():
            if name == "LARA":
                return model.head(model.coarse_decoder(z))
            if name == "GeneralizedAutoencoder":
                return model.decoder(z)
            return model.decoder(z).unsqueeze(1)

    published = pd.read_csv(PUBLISHED_CSV)
    models = MODELS
    crs = EXPECTED_CRS
    bits_list = BITS
    if args.smoke:
        models, crs, bits_list = ["LARA"], [10, 100], [4, 3, 2]
        print("\n[smoke] LARA only, CR 10 and 100, bits 4/3/2")

    # ------------------------------------------------- guard 2 + unquantized ref
    print("\n" + "=" * 74)
    print("GUARD 2 - reloaded checkpoints must reproduce the published metrics")
    print("=" * 74)
    unquant = {}
    for name in models:
        for cr in crs:
            model = build(name, cr)
            model.load_state_dict(torch.load(os.path.join(CKPT_DIR, f"best_{name}_cr{cr}.pth"),
                                             map_location="cpu", weights_only=True))
            model.eval()
            m = evaluate_model(model, test_data, cr, name)
            row = published[(published["Model"] == name) & (published["CR"] == cr)]
            pub = float(row["SNR"].iloc[0])
            d = abs(m["SNR"] - pub)
            ok = d <= GATE_TOL_DB
            unquant[(name, cr)] = m["SNR"]
            print(f"  {name:<22} CR={cr:<4} published {pub:7.3f}  recomputed {m['SNR']:7.3f}  "
                  f"d={d:6.4f} dB  {'OK' if ok else 'MISMATCH'}")
            if not ok:
                raise SystemExit(f"ABORT: {name} CR={cr} does not reproduce the published run")
    print(f"[guard 2] PASS - worst deviation "
          f"{max(abs(unquant[k] - float(published[(published['Model'] == k[0]) & (published['CR'] == k[1])]['SNR'].iloc[0])) for k in unquant):.5f} dB")

    # ------------------------------------------------------------- the pipeline
    print("\n" + "=" * 74)
    print("QUANTIZE + ENTROPY CODE")
    print("=" * 74)
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []
    t0 = time.time()

    for name in models:
        for cr in crs:
            lat = N_SAMPLES // cr
            model = build(name, cr)
            model.load_state_dict(torch.load(os.path.join(CKPT_DIR, f"best_{name}_cr{cr}.pth"),
                                             map_location="cpu", weights_only=True))
            model.eval()

            z_cal = encode_latent(model, name, calib_x).numpy().astype(np.float64)
            z_test = encode_latent(model, name, torch.tensor(test_data)).numpy().astype(np.float64)
            assert z_test.shape[1] == lat, f"{name} CR={cr}: latent {z_test.shape[1]} != {lat}"

            lo = z_cal.min(axis=0)
            hi = z_cal.max(axis=0)
            span = np.where(hi - lo > 0, hi - lo, 1.0)
            n_par = sum(p.numel() for p in model.parameters())
            model_bits = n_par * 32
            side_bits = lat * 2 * 16          # per-dim min/max as float16, sent once

            print(f"\n  {name} CR={cr}  latent={lat}  params={n_par:,}  "
                  f"uncoded BPS={32.0 * lat / N_SAMPLES:.4f}")

            for b in bits_list:
                A = 1 << b
                delta = span / (A - 1)
                q = np.clip(np.rint((z_test - lo) / delta), 0, A - 1).astype(np.int64)

                # measure() encodes every trace, and decodes each one back to
                # confirm the symbols are recoverable before the rate is used
                rates = measure(q, b, A, verify=True)
                z_rec = lo + q * delta

                bits_per_trace = rates["coded_bits"]

                rec = decode_latent(model, name,
                                    torch.tensor(z_rec, dtype=torch.float32)).numpy()
                if rec.ndim == 3:
                    rec = rec[:, 0, :]

                snrs, ssims, mses = [], [], []
                for t in range(rec.shape[0]):
                    o, r = test_data_np[t], rec[t]
                    mse = float(np.mean((o - r) ** 2))
                    snrs.append(10 * math.log10(float(np.mean(o ** 2)) / (mse + 1e-12))
                                if mse > 0 else float("inf"))
                    ssims.append(float(calculate_ssim_1d(o, r)))
                    mses.append(mse)

                def bps_for(trace_bits):
                    return (model_bits + side_bits + trace_bits) / (z_test.shape[0] * N_SAMPLES)

                rows.append({
                    "Model": name, "CR": cr, "bits": b, "latent_dim": lat,
                    "coder_bits_per_trace": round(bits_per_trace, 2),
                    "entropy_bound_bits_per_trace": round(rates["entropy_bound_bits"], 2),
                    "fixed_bits_per_trace": rates["fixed_bits"],
                    "side_info_bits": side_bits,
                    "model_bits": model_bits,
                    "BPS_coded": round(bps_for(bits_per_trace), 4),
                    "BPS_entropy_bound": round(bps_for(rates["entropy_bound_bits"]), 4),
                    "BPS_uncoded": round(bps_for(lat * 32.0), 4),
                    "BPS_infinite_N": round(bits_per_trace / N_SAMPLES, 4),
                    "SNR_dB": round(float(np.mean(snrs)), 4),
                    "SSIM": round(float(np.mean(ssims)), 4),
                    "MSE": round(float(np.mean(mses)), 8),
                    "SNR_unquantized_dB": round(unquant[(name, cr)], 4),
                    "SNR_loss_dB": round(unquant[(name, cr)] - float(np.mean(snrs)), 4),
                })
                r = rows[-1]
                print(f"      b={b:<2} coded {bits_per_trace:>9.1f}  bound {r['entropy_bound_bits_per_trace']:>9.1f}"
                      f"  fixed {r['fixed_bits_per_trace']:>8.0f}  |  BPS {r['BPS_coded']:>8.4f}"
                      f"  (uncoded {r['BPS_uncoded']:.4f})  SNR {r['SNR_dB']:>7.3f}"
                      f"  loss {r['SNR_loss_dB']:>6.3f} dB")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, "bps_measurements.csv"), index=False)
    print(f"\nwrote bps_measurements.csv  ({len(df)} rows) in {(time.time() - t0) / 60:.1f} min")

    # ------------------------------------------- matched-quality operating points
    print("\n" + "=" * 74)
    print("MATCHED-QUALITY SUMMARY (lowest BPS within delta dB of the unquantized SNR)")
    print("=" * 74)
    summary_rows = []
    for (name, cr), g in df.groupby(["Model", "CR"]):
        base = float(g["SNR_unquantized_dB"].iloc[0])
        row = {"Model": name, "CR": int(cr), "SNR_unquantized_dB": round(base, 4),
               "BPS_uncoded_latent": round(32.0 * (N_SAMPLES // int(cr)) / N_SAMPLES, 4)}
        for delta in (0.25, 0.5, 1.0):
            ok = g[g["SNR_dB"] >= base - delta]
            if len(ok):
                best = ok.loc[ok["BPS_infinite_N"].idxmin()]
                row[f"bits@{delta}"] = int(best["bits"])
                row[f"BPS_latent@{delta}"] = best["BPS_infinite_N"]
                row[f"BPS_total_N200@{delta}"] = best["BPS_coded"]
                row[f"SNR@{delta}"] = best["SNR_dB"]
            else:
                row[f"bits@{delta}"] = None
                row[f"BPS_latent@{delta}"] = None
                row[f"BPS_total_N200@{delta}"] = None
                row[f"SNR@{delta}"] = None
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows).sort_values(["Model", "CR"])
    summary.to_csv(os.path.join(OUT_DIR, "bps_summary.csv"), index=False)
    print(summary.to_string(index=False))

    # ------------------------------------------------------------- amortization
    amort_rows = []
    for _, r in df.iterrows():
        for n_tr in (200, 1000, 10_000, 100_000, 1_000_000):
            bps = (r["model_bits"] + r["side_info_bits"]
                   + n_tr * r["coder_bits_per_trace"]) / (n_tr * N_SAMPLES)
            amort_rows.append({
                "Model": r["Model"], "CR": int(r["CR"]), "bits": int(r["bits"]),
                "N_traces": n_tr,
                "BPS_total": round(bps, 4),
                "model_share_pct": round(100.0 * r["model_bits"]
                                         / (n_tr * r["coder_bits_per_trace"]
                                            + r["model_bits"] + r["side_info_bits"]), 2),
            })
    amort = pd.DataFrame(amort_rows)
    amort.to_csv(os.path.join(OUT_DIR, "bps_amortization.csv"), index=False)
    print("\nBPS_total for LARA at CR=10 across N (b=4):")
    sel = amort[(amort.Model == "LARA") & (amort.CR == 10) & (amort.bits == 4)]
    for _, r in sel.iterrows():
        print(f"    N={int(r.N_traces):>9,}  BPS {r.BPS_total:>10.4f}  "
              f"model share {r.model_share_pct:>6.2f}%")

    # -------------------------------------------------------------- paper table
    tex = [r"\begin{tabular}{llrrr}", r"\toprule",
           r"CR & Model & latent dims & BPS (uncoded) & BPS (quantized+coded) \\", r"\midrule"]
    for _, r in summary.iterrows():
        coded_bps = r["BPS_latent@0.5"]
        if coded_bps is not None and not pd.isna(coded_bps):
            tex.append(f"{int(r['CR'])} & {r['Model']} & {N_SAMPLES // int(r['CR'])} & "
                       f"{r['BPS_uncoded_latent']:.3f} & {coded_bps:.3f} \\\\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(OUT_DIR, "bps_paper_table.tex"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(tex) + "\n")
    with open(os.path.join(OUT_DIR, "bps_paper_table.md"), "w", encoding="utf-8") as fh:
        cols = list(summary.columns)
        fh.write("| " + " | ".join(cols) + " |\n")
        fh.write("|" + "|".join(["---"] * len(cols)) + "|\n")
        for _, r in summary.iterrows():
            fh.write("| " + " | ".join("" if pd.isna(r[c]) else str(r[c]) for c in cols) + " |\n")

    # ------------------------------------------------------------------ figures
    COL = {"LARA": "#66c2a5", "GeneralizedAutoencoder": "#8da0cb", "AE_PureConcat": "#80b1d3"}
    MKS = {"LARA": "o", "GeneralizedAutoencoder": "s", "AE_PureConcat": "^"}
    draw_rd_figures(df, models, crs, MKS, COL)
    draw_bps_vs_cr(summary, models, crs, MKS, COL)

    print(f"\nall outputs written to {OUT_DIR}")
    for f in sorted(os.listdir(OUT_DIR)):
        print(f"  {os.path.getsize(os.path.join(OUT_DIR, f)):>9} {f}")
    return df


# =============================================================== figure code
# Colour encodes the quantizer bit depth, not the model: each panel is already
# titled with its model, and spending a second colour channel on CR as well
# would be unreadable. The ten CR curves are thinned to RD_CRS and drawn as thin
# grey trajectories, each labelled once at its right-hand end.
RD_CRS = [2, 5, 10, 50, 100]
CMAP = "viridis"


def _b_colours(bits_list):
    """Discrete colour per quantizer depth: light = coarse, dark = fine."""
    ordered = sorted(bits_list)
    cmap = plt.get_cmap(CMAP)
    n = len(ordered)
    return {b: cmap(0.12 + 0.76 * i / max(n - 1, 1)) for i, b in enumerate(ordered)}, ordered


def draw_rd_figures(df, models, crs, MKS, COL):
    bcol, ordered_b = _b_colours(sorted(df["bits"].unique()))

    for suffix, logx, logy in (("", False, False), ("_loglog", True, True)):
        # CR=50 and CR=100 have near-identical endpoints on a linear axis, so
        # labelling all five there would recreate the crowding this figure is
        # meant to remove. The log axis separates them, so both are labelled.
        label_crs = RD_CRS if logx else [c for c in RD_CRS if c <= 10]
        fig, axes = plt.subplots(1, 3, figsize=(19, 6.4), sharey=logy)
        for ax, name in zip(axes, models):
            g = df[df["Model"] == name]
            # rank the curves by their low-rate endpoint so the CR labels can be
            # fanned out vertically; CR=50 and CR=100 otherwise sit on top of
            # each other on a linear axis
            ends = {cr: g[g["CR"] == cr].sort_values("bits", ascending=False).iloc[-1]
                    for cr in RD_CRS if not g[g["CR"] == cr].empty}
            rank = {cr: i for i, cr in
                    enumerate(sorted(ends, key=lambda c: ends[c]["BPS_infinite_N"]))}
            stagger = [0, 11, -11, 22, -22]
            for cr in RD_CRS:
                h = g[g["CR"] == cr].sort_values("bits", ascending=False)
                if h.empty:
                    continue
                ax.plot(h["BPS_infinite_N"], h["SNR_dB"], "-", color="#cccccc",
                        linewidth=0.9, zorder=1)
                for b, grp in h.groupby("bits"):
                    ax.plot(grp["BPS_infinite_N"], grp["SNR_dB"], ls="none",
                            marker=MKS[name], color=bcol[int(b)], markersize=7,
                            markeredgecolor="black", markeredgewidth=0.5, zorder=3)
                # one CR label per curve, at the low-rate end, fanned out
                if cr in label_crs:
                    last = h.iloc[-1]
                    ax.annotate(f"CR={cr}", (last["BPS_infinite_N"], last["SNR_dB"]),
                                textcoords="offset points",
                                xytext=(5, stagger[rank[cr] % len(stagger)]),
                                ha="left", va="center", fontsize=7.5, color="#666666",
                                zorder=4)
            if logx:
                ax.set_xscale("log")
            if logy:
                ax.set_yscale("log")
                # keep the dB axis reading as plain numbers, not 2x10^1
                ax.yaxis.set_major_formatter(
                    matplotlib.ticker.ScalarFormatter())
                ax.yaxis.set_minor_formatter(
                    matplotlib.ticker.NullFormatter())
            ax.set_xlabel("BPS  (latent only, model amortised)", fontsize=11)
            ax.set_title(name, fontsize=13, fontweight="bold")
            ax.grid(True, alpha=0.3, which="both")
        axes[0].set_ylabel("Reconstruction SNR (dB)", fontsize=11)

        handles = [plt.Line2D([], [], ls="none", marker="o", color=bcol[b],
                              markeredgecolor="black", markeredgewidth=0.5,
                              markersize=7, label=f"b={b}") for b in ordered_b]
        fig.legend(handles=handles, loc="upper center", ncol=len(ordered_b),
                   fontsize=10, frameon=True, title="quantizer precision (bits)",
                   title_fontsize=10, bbox_to_anchor=(0.5, 1.005))
        tag = "Rate-Distortion, log-log" if logx else "Rate-Distortion, linear axes"
        fig.suptitle(f"{tag}: quantize + adaptive arithmetic code, "
                     f"CR in {RD_CRS}", fontsize=14, fontweight="bold", y=1.10)
        fig.tight_layout(rect=(0, 0, 1, 0.90))
        base = os.path.join(OUT_DIR, f"bps_rd_curves{suffix}")
        for ext in ("png", "eps"):
            fig.savefig(f"{base}.{ext}", dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote bps_rd_curves{suffix}.png / .eps  "
              f"(x={'log' if logx else 'linear'}, y={'log' if logy else 'linear'}, "
              f"{len(RD_CRS)} CR x {len(ordered_b)} depths = "
              f"{len(RD_CRS) * len(ordered_b)} markers/panel)")


def draw_bps_vs_cr(summary, models, crs, MKS, COL):
    fig, ax = plt.subplots(figsize=(11, 7))
    for name in models:
        h = summary[summary["Model"] == name].sort_values("CR")
        ax.semilogx(h["CR"], h["BPS_latent@0.5"], marker=MKS[name], linestyle="-",
                    color=COL[name], label=f"{name} (coded, <=0.5 dB loss)",
                    markersize=8, markeredgecolor="black", linewidth=2)
    crs_a = np.array(crs, dtype=float)
    ax.semilogx(crs_a, 32.0 / crs_a, "k--", linewidth=1.6,
                label=r"uncoded float32 latent  ($32/\mathrm{CR}$)")
    ax.set_xlabel("Compression Ratio", fontsize=13)
    ax.set_ylabel("BPS  (latent only, model amortised)", fontsize=13)
    ax.set_title("Effective BPS vs Compression Ratio", fontsize=15, fontweight="bold")
    ax.set_xticks(crs_a)
    ax.set_xticklabels([str(c) for c in crs_a])
    ax.minorticks_off()
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc="upper right")
    fig.tight_layout()
    for ext in ("png", "eps"):
        fig.savefig(os.path.join(OUT_DIR, f"bps_vs_cr.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


# ===================================================================== entry
MODELS_ALL = ["LARA", "GeneralizedAutoencoder", "AE_PureConcat"]
MKS = {"LARA": "o", "GeneralizedAutoencoder": "s", "AE_PureConcat": "^"}
COL = {"LARA": "#66c2a5", "GeneralizedAutoencoder": "#8da0cb", "AE_PureConcat": "#80b1d3"}


def figures_only():
    """Redraw the figures from bps_measurements.csv / bps_summary.csv.

    Used after a plotting change so the figures can be regenerated in seconds
    without re-running the 13-minute measurement pass.
    """
    mfile = os.path.join(OUT_DIR, "bps_measurements.csv")
    sfile = os.path.join(OUT_DIR, "bps_summary.csv")
    for f in (mfile, sfile):
        if not os.path.exists(f):
            raise SystemExit(f"ABORT: {f} not found; run the full analysis first")
    df = pd.read_csv(mfile)
    summary = pd.read_csv(sfile)
    models = [m for m in MODELS_ALL if m in set(df["Model"])]
    crs = sorted(df["CR"].unique().tolist())
    print(f"figures-only: {len(df)} measurement rows, CRs {crs}, models {models}")
    draw_rd_figures(df, models, crs, MKS, COL)
    # bps_vs_cr is only redrawn on explicit request. Matplotlib stamps
    # %%CreationDate into EPS, so rewriting it would change those bytes even
    # though the figure is identical.
    if "--with-bps-vs-cr" in sys.argv:
        draw_bps_vs_cr(summary, models, crs, MKS, COL)
    else:
        print("  bps_vs_cr left untouched (pass --with-bps-vs-cr to redraw it)")
    print(f"\noutput: {OUT_DIR}")
    for f in sorted(os.listdir(OUT_DIR)):
        print(f"  {os.path.getsize(os.path.join(OUT_DIR, f)):>9} {f}")


if __name__ == "__main__":
    if "--figures-only" in sys.argv:
        figures_only()
    else:
        main()
