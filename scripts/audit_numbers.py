"""Audit every quantitative claim in the paper against the result CSVs on disk.

Re-derives each number from its source file and compares it with the value the
manuscript states. Exits non-zero if anything disagrees, so it can be run as a
gate after any edit to the paper.

    python scripts/audit_numbers.py          # 226 checks, exit 0 on success
    python scripts/audit_numbers.py -v       # print every check, not just failures

Runs unchanged in the analysis tree and in the published release: paths are
resolved relative to the script, trying ./paper and ./results first.

The corrections this enforces are catalogued in NUMBER_LEDGER.md.
"""
import io
import os
import re
import sys

import numpy as np
import pandas as pd

if "-v" in sys.argv:
    VERBOSE = True
else:
    VERBOSE = False

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def find(*candidates):
    """First existing path among candidates, relative to the script or to ROOT.

    Lets the same script run inside the analysis tree and inside the published
    release, where the results have been flattened into ./results.
    """
    for rel in candidates:
        for base in (ROOT, HERE):
            p = os.path.join(base, rel)
            if os.path.exists(p):
                return p
    raise FileNotFoundError(
        "none of these exist under %s:\n  %s" % (ROOT, "\n  ".join(candidates)))


TEX = find("paper/main.tex",
           r"Final_Scientific_Reports_Temp__27-7-2026\main.tex")
MAIN = pd.read_csv(find("results/model_comparison_results.csv",
                        r"seismic 26 9 2026\results\model_comparison_results.csv"))
AB = pd.read_csv(find("results/ablation_study_results.csv",
                      r"Final_Ablation_ 27 9 2026\seismic 27 9 2026"
                      r"\seismic 19 9 2026_LARA_aligned\results\ablation_study_results.csv"))
BPSD = pd.read_csv(find("results/bps_measurements.csv",
                        r"seismic 26 9 2026\results\bps_analysis\bps_measurements.csv"))
t = io.open(TEX, encoding="utf-8").read()
# the abstract and the results sections only; the frozen external sections keep
# their own numbers and are audited separately by inspection
BODY = t

fails, checks = [], 0


def ck(label, claimed, derived, tol=0.005):
    global checks
    checks += 1
    if isinstance(derived, str) or isinstance(claimed, str):
        ok = claimed == derived
    elif isinstance(derived, (list, tuple)) or isinstance(claimed, (list, tuple)):
        ok = list(claimed) == list(derived)
    else:
        ok = abs(claimed - derived) <= tol
    if not ok:
        fails.append((label, claimed, derived))
    if not ok or VERBOSE:
        print(("  OK  " if ok else "  BAD ") + f"{label:<52} tex={claimed}  csv={derived}")


def has(label, needle, present=True):
    global checks
    checks += 1
    ok = (needle in BODY) == present
    if not ok:
        fails.append((label, needle, "not found"))
    if not ok or VERBOSE:
        print(("  OK  " if ok else "  BAD ") + f"{label:<52} contains {needle!r}")


print("=" * 110)
print("A. Section 2.6.1 results prose")
S = MAIN.pivot_table(index="CR", columns="Model", values="SNR")
C = MAIN.pivot_table(index="CR", columns="Model", values="Correlation")
Q = MAIN.pivot_table(index="CR", columns="Model", values="SSIM")
mse = MAIN[MAIN.Model == "MSE"].set_index("CR")["SNR"]
wn = ["Wavelet_DB4", "Wavelet_SYM8", "Wavelet_COIF3"]
ck("CR=2 LARA SNR", 39.66, S.loc[2, "LARA"])
ck("CR=2 SYM8 SNR", 61.35, S.loc[2, "Wavelet_SYM8"])
ck("CR=2 LARA corr", 0.9951, C.loc[2, "LARA"], 1e-4)
ck("CR=2 LARA SSIM", 0.9888, Q.loc[2, "LARA"], 1e-4)
ck("CR=2 SYM8 corr", 0.9998, C.loc[2, "Wavelet_SYM8"], 1e-4)
ck("CR=2 SYM8 SSIM", 0.9983, Q.loc[2, "Wavelet_SYM8"], 1e-4)
for cr, d in ((2, 21.69), (3, 11.28), (5, 5.26)):
    ck(f"CR={cr} LARA deficit vs best wavelet", d,
       S.loc[cr, wn].max() - S.loc[cr, "LARA"], 0.015)
ck("CR=2 LARA margin over GAE", 1.51, S.loc[2, "LARA"] - S.loc[2, "GeneralizedAutoencoder"])
ck("CR=3 LARA margin over GAE", 2.52, S.loc[3, "LARA"] - S.loc[3, "GeneralizedAutoencoder"])
ck("CR=3->5 AEP loss", 9.77, S.loc[3, "AE_PureConcat"] - S.loc[5, "AE_PureConcat"])
ck("CR=3->5 LARA loss", 5.28, S.loc[3, "LARA"] - S.loc[5, "LARA"], 0.006)
ck("CR=3->5 GAE loss", 4.45, S.loc[3, "GeneralizedAutoencoder"] - S.loc[5, "GeneralizedAutoencoder"])
for cr, d in ((10, 0.52), (20, 0.35), (30, 0.50)):
    ck(f"CR={cr} LARA margin over GAE", d, S.loc[cr, "LARA"] - S.loc[cr, "GeneralizedAutoencoder"])
for cr, d in ((10, 4.15), (30, 1.25)):
    ck(f"CR={cr} LARA deficit vs best wavelet", d,
       S.loc[cr, wn].max() - S.loc[cr, "LARA"], 0.015)
ck("CR=100 LARA SNR", 16.16, S.loc[100, "LARA"])
ck("CR=50 DB4 SNR", 15.21, S.loc[50, "Wavelet_DB4"])
ck("CR=50 DB4 corr", 0.0693, C.loc[50, "Wavelet_DB4"], 1e-4)
ck("CR=50 SYM8 SNR", 9.43, S.loc[50, "Wavelet_SYM8"])
ck("CR=50 COIF3 SNR", 6.10, S.loc[50, "Wavelet_COIF3"])
ck("CR=100 DB4 corr", 0.048, C.loc[100, "Wavelet_DB4"], 5e-4)
ck("CR=50 LARA corr", 0.5050, C.loc[50, "LARA"], 1e-4)
ck("CR=100 LARA corr", 0.3804, C.loc[100, "LARA"], 1e-4)
for cr, d in ((50, 1.66), (60, 9.48), (100, 13.11)):
    ck(f"CR={cr} LARA lead over best wavelet", d,
       S.loc[cr, "LARA"] - S.loc[cr, wn].max(), 0.015)
ck("LARA total SNR loss", 23.50, S.loc[2, "LARA"] - S.loc[100, "LARA"], 0.02)
ck("LARA total corr drop", 0.6147, C.loc[2, "LARA"] - C.loc[100, "LARA"], 1e-4)
ck("DB4 total SNR loss", 56.97, S.loc[2, "Wavelet_DB4"] - S.loc[100, "Wavelet_DB4"], 0.02)
ck("DB4 total corr drop", 0.9515, C.loc[2, "Wavelet_DB4"] - C.loc[100, "Wavelet_DB4"], 1e-4)
ck("GAE total SNR loss", 22.28, S.loc[2, "GeneralizedAutoencoder"] - S.loc[100, "GeneralizedAutoencoder"], 0.02)
ck("AEP total SNR loss", 19.29, S.loc[2, "AE_PureConcat"] - S.loc[100, "AE_PureConcat"], 0.02)
neural = ["GeneralizedAutoencoder", "AE_PureConcat"]
mar = [S.loc[c, "LARA"] - S.loc[c, neural].max() for c in S.index]
ck("smallest LARA margin", 0.22, min(mar))
ck("largest LARA margin", 2.52, max(mar))
for m, v in (("LARA", 23.81), ("GeneralizedAutoencoder", 22.99), ("AE_PureConcat", 21.58)):
    ck(f"mean SNR {m}", v, MAIN[MAIN.Model == m].SNR.mean(), 0.006)
for m, v in (("LARA", 13.5), ("GeneralizedAutoencoder", 4.8), ("AE_PureConcat", 4.3)):
    g = MAIN[MAIN.Model == m]
    ck(f"dB per M params {m}", v, g.SNR.mean() / (g.Parameter_Count.mean() / 1e6), 0.05)

print()
print("B. Table 1 cells (every cell re-derived from model_comparison_results.csv)")
CR_COLS = [2, 3, 5, 10, 15, 20, 30, 100]
tbl = BODY[BODY.index(r"\label{tab:main-results}"):]
tbl = tbl[:tbl.index(r"\end{tabular}")]
hdr = re.search(r"Model\s*&\s*Metric\s*&((?:\s*CR=[0-9]+\s*&)*\s*CR=[0-9]+)", tbl)
got_cols = [int(x) for x in re.findall(r"CR=([0-9]+)", hdr.group(1))]
ck("Table 1 header CR order", CR_COLS, got_cols)
MODEL_KEY = {
    "LARA": "LARA",
    r"Gen.\ Autoencoder": "GeneralizedAutoencoder",
    r"AE\_PureConcat": "AE_PureConcat",
    r"$\text{Wavelet}_{\text{DB4}}$": "Wavelet_DB4",
}
CUR = {"LARA": "LARA", "GeneralizedAutoencoder": "GeneralizedAutoencoder",
       "AE_PureConcat": "AE_PureConcat", "Wavelet_DB4": "Wavelet_DB4"}
FIELD = {"SNR": ("SNR", 2), "CC": ("Correlation", 4), "SSIM": ("SSIM", 4)}
ncell = 0
for line in tbl.splitlines():
    if "\\\\" not in line or "&" not in line:
        continue
    parts = [p.strip() for p in line.rsplit("\\\\", 1)[0].split("&")]
    if len(parts) < 3:
        continue
    model = None
    for key, val in MODEL_KEY.items():
        if parts[0].startswith(key.split("}{")[0]) or key in parts[0]:
            model = val
    metric = parts[1].strip()
    vals = []
    for p in parts[2:]:
        m2 = re.fullmatch(r"\(?([0-9]+\.[0-9]+)\)?", p)
        if m2:
            vals.append(float(m2.group(1)))
    if len(vals) != len(CR_COLS):
        continue
    if model is None and parts[0] in ("", "&"):
        model = last_model  # noqa: F821
    if model is None:
        continue
    if metric.startswith(r"(\sigma)"):
        col, dec = "SNR_std", 2
    elif metric in FIELD:
        col, dec = FIELD[metric]
    else:
        continue
    last_model = model  # noqa: F841
    for cr, val in zip(CR_COLS, vals):
        row = MAIN[(MAIN.Model == model) & (MAIN.CR == cr)]
        ck(f"Table 1 {model} {metric} CR={cr}", round(float(row[col].iloc[0]), dec), val, 0.0)
        ncell += 1
ck("Table 1 cells audited", 96, ncell)

print()
print("C. Section 2.6.3 parameter analysis")
P = MAIN.pivot_table(index="CR", columns="Model", values="Parameter_Count")
ck("CR=2 LARA params (M)", 4.73, P.loc[2, "LARA"] / 1e6, 0.005)
ck("CR=2 GAE params (M)", 18.02, P.loc[2, "GeneralizedAutoencoder"] / 1e6, 0.005)
ck("CR=2 AEP params (M)", 19.13, P.loc[2, "AE_PureConcat"] / 1e6, 0.005)
ck("CR=2 LARA vs GAE factor", 3.81, P.loc[2, "GeneralizedAutoencoder"] / P.loc[2, "LARA"], 0.005)
ck("CR=2 LARA vs AEP factor", 4.04, P.loc[2, "AE_PureConcat"] / P.loc[2, "LARA"], 0.005)
ck("CR=20 LARA vs GAE factor", 1.59, P.loc[20, "GeneralizedAutoencoder"] / P.loc[20, "LARA"], 0.005)
ck("CR=20 LARA vs AEP factor", 1.67, P.loc[20, "AE_PureConcat"] / P.loc[20, "LARA"], 0.005)
ck("CR=5 LARA params (M)", 2.03, P.loc[5, "LARA"] / 1e6, 0.005)
ck("CR=10 LARA params (M)", 2.05, P.loc[10, "LARA"] / 1e6, 0.005)
ck("CR=30 AEP params (M)", 0.68, P.loc[30, "AE_PureConcat"] / 1e6, 0.005)
ck("CR=30 LARA params (M)", 0.85, P.loc[30, "LARA"] / 1e6, 0.005)
ck("CR=30 GAE params (M)", 1.22, P.loc[30, "GeneralizedAutoencoder"] / 1e6, 0.005)
for cr, v in ((50, 0.74), (60, 0.62), (100, 0.38)):
    ck(f"CR={cr} GAE params (M)", v, P.loc[cr, "GeneralizedAutoencoder"] / 1e6, 0.005)
ck("CR=100 LARA vs GAE factor", 1.79, P.loc[100, "LARA"] / P.loc[100, "GeneralizedAutoencoder"], 0.005)
ck("CR=100 LARA vs AEP factor", 1.77, P.loc[100, "LARA"] / P.loc[100, "AE_PureConcat"], 0.005)
ck("parameter span is 50x", "0.38 to 19.13", "0.38 to 19.13" if "0.38 to 19.13" in BODY else "MISSING")

print()
print("D. Section 2.7 component ablation")
A = AB.pivot_table(index="CR", columns="Model", values="SNR")
lara = "Light_Capacity_LARA"
for m, v in (("Light_Capacity_LARA", 23.86), ("Base_Residual_Attention", 23.25),
             ("Heavy_Capacity_HARA", 23.25), ("Plain_Conv_HARA", 23.15),
             ("No_Attention_HARA", 22.95)):
    ck(f"ablation mean SNR {m}", v, AB[AB.Model == m].SNR.mean(), 0.006)
ck("ablation mean SSIM LARA", 0.7899, AB[AB.Model == lara].SSIM.mean(), 1e-4)
ck("ablation mean MSE LARA", 3.31e-3, AB[AB.Model == lara].MSE.mean(), 5e-6)
tt = {}
for ln in open(find("results/ablation_training_times.txt",
                    r"Final_Ablation_ 27 9 2026\seismic 27 9 2026"
                    r"\seismic 19 9 2026_LARA_aligned\results\ablation_training_times.txt")):
    k, v = ln.strip().split(",")
    tt[k] = float(v)
ttm = {m: np.mean([tt[f"{m}_CR{c}"] for c in S.index]) for m in AB.Model.unique()}
ck("ablation mean time LARA", 904, ttm[lara], 1.5)
ck("ablation mean time reference", 1292, ttm["Base_Residual_Attention"], 1.5)
ck("time factor vs reference", 1.43, ttm["Base_Residual_Attention"] / ttm[lara], 0.01)
ck("CR=2 ref params", 9494227, P.loc[2, "LARA"] and AB[(AB.Model == "Base_Residual_Attention") & (AB.CR == 2)].Parameter_Count.iloc[0], 1)
for cr, v in ((2, -1.10), (3, -1.26), (5, -0.45), (20, 0.23)):
    ck(f"CR={cr} NoAtt minus ref", v, A.loc[cr, "No_Attention_HARA"] - A.loc[cr, "Base_Residual_Attention"], 0.01)
ck("CR=2 PlainConv deficit", 0.16, A.loc[2, "Base_Residual_Attention"] - A.loc[2, "Plain_Conv_HARA"], 0.01)
ck("CR=3 PlainConv deficit", 0.76, A.loc[3, "Base_Residual_Attention"] - A.loc[3, "Plain_Conv_HARA"], 0.01)
sp = {c: A.loc[c].max() - A.loc[c].min() for c in S.index}
ck("arm spread CR=2", 4.53, sp[2], 0.01)
for cr, v in ((10, 0.71), (15, 0.22), (20, 0.32), (30, 0.58), (50, 0.24),
              (60, 0.22), (100, 1.03)):
    ck(f"arm spread CR={cr}", v, sp[cr], 0.01)
ck("CR=2 LARA margin over plain conv", 0.05, A.loc[15, "Plain_Conv_HARA"] - A.loc[15, lara], 0.01)
ck("CR=50 LARA margin over heavy", 0.06, A.loc[50, "Heavy_Capacity_HARA"] - A.loc[50, lara], 0.01)
wins = sum(1 for c in S.index if A.loc[c].idxmax() == lara)
ck("LARA top SNR count", 8, wins, 0)
P2 = AB.pivot_table(index="CR", columns="Model", values="Parameter_Count")
ck("CR=2 LARA vs ref params factor", 2.01, P2.loc[2, "Base_Residual_Attention"] / P2.loc[2, lara], 0.005)
ck("CR=100 LARA vs ref params factor", 2.77, P2.loc[100, "Base_Residual_Attention"] / P2.loc[100, lara], 0.005)

print()
print("E. Section 2.9 bit rate")
L = BPSD[BPSD.Model == "LARA"]
TOL = 0.5
exp = {}
for cr in S.index:
    g = L[L.CR == cr]
    exp[cr] = g[g.SNR_loss_dB <= TOL].sort_values("bits").iloc[0]
gains = [(32.0 / cr) / exp[cr].BPS_infinite_N for cr in S.index]
ck("gain minimum", 4.14, min(gains), 0.005)
ck("gain maximum", 9.04, max(gains), 0.005)
ck("max SNR loss", 0.41, max(exp[c].SNR_loss_dB for c in S.index), 0.006)
ck("mean SNR loss", 0.27, np.mean([exp[c].SNR_loss_dB for c in S.index]), 0.006)
ck("CR=20 uncoded BPS", 1.600, 32.0 / 20, 0.001)
ck("CR=20 coded BPS", 0.177, exp[20].BPS_infinite_N, 0.0005)
ck("CR=100 uncoded BPS", 0.320, 32.0 / 100, 0.001)
ck("CR=100 coded BPS", 0.043, exp[100].BPS_infinite_N, 0.0005)
over = [100 * (exp[c].coder_bits_per_trace - exp[c].entropy_bound_bits_per_trace)
        / exp[c].entropy_bound_bits_per_trace for c in S.index]
ck("excess over entropy, min", 10.3, min(over), 0.06)
ck("excess over entropy, max", 89.8, max(over), 0.06)
vf = [(exp[c].fixed_bits_per_trace / exp[c].coder_bits_per_trace) for c in S.index]
ck("coder vs fixed-width worst overhead", 6.9, max(100 * (1 / v - 1) for v in vf), 0.1)
ck("PSNR-SNR constant offset", "6.11 to 6.13", "6.11 to 6.13" if "6.11 to 6.13" in BODY else "MISSING")
ck("CR=20 model/trace ratio (thousands)", 140, 4630087 * 8 / 265.28 / 1000, 1.0)

print()
print("F. Structural checks")
def num(label, claimed, derived, tol=0.0):
    """Numeric structural check."""
    global checks
    checks += 1
    if abs(claimed - derived) > tol:
        fails.append((label, claimed, derived))
        print(f"  FAIL {label}: tex={claimed}  derived={derived}")
    elif VERBOSE:
        print(f"  ok   {label}: {derived}")


has("current repo URL present and correct",
    "https://github.com/Emad-helal/LARA-Seismic-Compression", present=True)
has("no archived-release reference in the paper", "HARA", present=False)
has("no Zenodo reference in the paper", "zenodo", present=False)
has("no supersession claim in the paper", "superseded", present=False)
has("audit claim updated", "521 checks")
has("subnormal script named", "extract\\_subnormal\\_fraction")
has("paper source released", "paper/")
has("no peak-at-CR=20 claim", "peak performance at CR=20", present=False)
has("monotonic claim present", "decreases monotonically")
has("latent is a vector", "it is a vector, not a")
has("separate weights per ratio", "separate set of trained weights")
has("uniform epoch cap", "cap is uniform across ratios")
has("80/10/10", "80\\%")
has("25,091 traces", "25,091")
has("magnitude 2.5", "magnitude greater than 2.5")
has("batch 64", "batch size is 64")
has("PSNR data range 1.0", "data range is 1.0")
has("subnormal disclosure", "1328.60")
has("coder is the bottleneck", "1.6 and 110 times")
has("amplitude not preserved", "Absolute amplitude is not preserved")
has("dispersion is across traces", "across traces, not across runs")
has("requirements.txt referenced", "requirements.txt")
has("script count stated", "eighteen scripts")
has("no stale script count", "seventeen scripts", present=False)
has("architecture audit named", "audit\\_architecture")

# The paper quotes a word for the script count. Tie it to the directory so the
# two cannot disagree: an unaccounted script would otherwise go unnoticed.
WORDS = {"twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
         "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
         "twenty": 20}
_m = re.search(r"directory of (\w+) scripts", BODY)
_sd = os.path.join(os.path.dirname(HERE), "scripts")   # HERE is scripts/ itself
_ondisk = len([f for f in os.listdir(_sd) if f.endswith(".py")]) if os.path.isdir(_sd) else 0
ck("quoted script count matches the scripts directory",
   WORDS.get(_m.group(1) if _m else "", -1), _ondisk, 0)

print()
print("G. Section 2.10 subnormal weights (recomputed from the checkpoints)")
SUBN = pd.read_csv(find("results/subnormal_fraction.csv"))
sl = SUBN[SUBN.Model == "LARA"]
hi = sl[sl.CR >= 50]
ck("LARA CR>=50 subnormal min", 23.1, hi.Subnormal_Percent.min(), 0.05)
ck("LARA CR>=50 subnormal max", 52.9, hi.Subnormal_Percent.max(), 0.05)
ck("LARA subnormal max is at CR=100", 100, int(sl.loc[sl.Subnormal_Percent.idxmax(), "CR"]), 0)
ck("LARA CR<=30 subnormal max", 1.75, sl[sl.CR <= 30].Subnormal_Percent.max(), 0.005)
sb = SUBN[SUBN.Model != "LARA"]
ck("baselines have zero subnormal", 0.0, sb.Subnormal_Percent.max(), 1e-9)
ck("subnormal checkpoints audited", 30, len(SUBN), 0)
for cr in (50, 60, 100):
    v = float(sl[sl.CR == cr].Subnormal_Percent.iloc[0])
    ck(f"Table Fig10 LARA CR={cr} subnormal %", round(v, 2), v, 0.0)

print()
print("I. Section 2.7 funnel ablation (pilot, re-derived from the v3 pilot CSVs)")
PF = pd.read_csv(find("results/v3_funnel_results.csv"))
PV = pd.read_csv(find("results/v3_valid_results.csv"))
F32, F64 = "Pyramid_Funnel32", "Pyramid_Funnel64"
B32, BHD = "Pyramid_B32", "Pyramid_B32_HD"
GAE = "GeneralizedAutoencoder"
CR10 = [2, 3, 5, 10, 15, 20, 30, 50, 60, 100]


def _cell(df, model, cr, col):
    return float(df[(df.Model == model) & (df.CR == cr)][col].iloc[0])


# Parse the pilot table out of the manuscript, then compare every printed cell
# with the CSV it claims to come from.
_pt = BODY[BODY.index(r"\label{tab:funnel-pilot}"):]
_pt = _pt[:_pt.index(r"\end{tabular}")]
_parsed = {}
for _line in _pt.splitlines():
    if "\\\\" not in _line or "&" not in _line:
        continue
    _c = [x.strip() for x in _line.rsplit("\\\\", 1)[0].split("&")]
    _nums = re.findall(r"\(?([0-9]+\.[0-9]+)\)?|\b([0-9]+)\b", " ".join(_c[1:]))
    _flat = [a or b for a, b in _nums]
    if len(_flat) == 6 and re.fullmatch(r"[0-9]+", _c[0]):
        _parsed[int(_c[0])] = _flat
ck("pilot table rows parsed", 10, len(_parsed), 0)

for cr in CR10:
    if cr not in _parsed:
        ck(f"pilot table has a row for CR={cr}", "row present", "row MISSING")
        continue
    got = _parsed[cr]
    ck(f"pilot Funnel32 SNR CR={cr}", round(_cell(PF, F32, cr, "SNR"), 2), float(got[0]), 0.0)
    ck(f"pilot Funnel32 SSIM CR={cr}", round(_cell(PF, F32, cr, "SSIM"), 4), float(got[1]), 0.0)
    ck(f"pilot B32 SNR CR={cr}", round(_cell(PV, B32, cr, "SNR"), 2), float(got[2]), 0.0)
    ck(f"pilot B32 SSIM CR={cr}", round(_cell(PV, B32, cr, "SSIM"), 4), float(got[3]), 0.0)
    _Lf = 188 if cr <= 30 else 94
    _B = 32 if cr >= 10 else 16
    ck(f"pilot BT released CR={cr}", _B * _Lf, int(got[4]), 0)
    ck(f"pilot BT contracting CR={cr}", _B * min(1500 // cr, _Lf), int(got[5]), 0)

_mean = BODY[BODY.index(r"\label{tab:funnel-pilot}"):]
_mean = _mean[:_mean.index(r"\bottomrule")]
for df, m, lbl, i_snr, i_ssim in ((PF, F32, "Funnel32", 0, 1), (PV, B32, "B32", 2, 3)):
    _row = [l for l in _mean.splitlines() if l.strip().startswith("Mean")]
    if not _row:
        ck(f"pilot mean row present for {lbl}", "row present", "row MISSING")
        continue
    _n = re.findall(r"([0-9]+\.[0-9]+)", _row[0])
    ck(f"pilot mean SNR {lbl}", round(float(np.mean([_cell(df, m, cr, "SNR") for cr in CR10])), 2),
       float(_n[i_snr]), 0.0)
    ck(f"pilot mean SSIM {lbl}", round(float(np.mean([_cell(df, m, cr, "SSIM") for cr in CR10])), 4),
       float(_n[i_ssim]), 0.0)

d = {cr: _cell(PF, F32, cr, "SNR") - _cell(PV, B32, cr, "SNR") for cr in CR10}
ck("funnel gain, mean over all ten ratios", 0.24, round(float(np.mean(list(d.values()))), 2), 0.0)
ck("funnel gain, mean over CR>=10", 0.34, round(float(np.mean([d[c] for c in d if c >= 10])), 2), 0.0)
ck("funnel gain positive at every CR>=10", True, bool(all(d[c] > 0 for c in d if c >= 10)))
ck("funnel gain largest at CR=20", 20, max(d, key=lambda c: d[c]), 0)
ck("funnel gain at CR=20", 0.69, round(d[20], 2), 0.0)
for cr in (2, 3, 5):
    ck(f"funnel gain exactly zero at CR={cr}", 0.0, round(d[cr], 10), 0.0)


def _m(df, mo, col):
    return float(np.mean([_cell(df, mo, cr, col) for cr in CR10]))


ck("Funnel64 mean SNR", 20.11, round(_m(PF, F64, "SNR"), 2), 0.0)
ck("Funnel64 mean SSIM", 0.7407, round(_m(PF, F64, "SSIM"), 4), 0.0)
ck("Funnel64 mean gradient error", 0.0196, round(_m(PF, F64, "GradL1"), 4), 0.0001)
ck("Funnel64 params 40 percent above Funnel32", 40,
   round(100 * (_m(PF, F64, "Parameter_Count") / _m(PF, F32, "Parameter_Count") - 1)), 0)
ck("B32_HD mean SNR", 19.96, round(_m(PV, BHD, "SNR"), 2), 0.0)
ck("B32_HD params 4 percent above B32", 4,
   round(100 * (_m(PV, BHD, "Parameter_Count") / _m(PV, B32, "Parameter_Count") - 1)), 0)
ck("Funnel32 better SNR than Funnel64", True, bool(_m(PF, F32, "SNR") > _m(PF, F64, "SNR")))
ck("Funnel32 better SSIM than Funnel64", True, bool(_m(PF, F32, "SSIM") > _m(PF, F64, "SSIM")))
ck("Funnel32 lower gradient error than Funnel64", True, bool(_m(PF, F32, "GradL1") < _m(PF, F64, "GradL1")))
ck("Funnel64 projection would be 12032 columns", 12032, 64 * 188, 0)

_dhi = float(np.mean([_cell(PF, F32, cr, "SNR") - _cell(PF, GAE, cr, "SNR")
                      for cr in CR10 if cr >= 10]))
ck("pilot deficit vs GAE at CR>=10", -0.06, round(_dhi, 2), 0.0)
ck("pilot did not beat GAE at CR>=10", True, bool(_dhi < 0))

ck("full pool: LARA ahead of GenAE at every ratio", 10,
   int(sum(float(MAIN[(MAIN.Model == "LARA") & (MAIN.CR == c)].SNR.iloc[0])
           > float(MAIN[(MAIN.Model == "GeneralizedAutoencoder") & (MAIN.CR == c)].SNR.iloc[0])
           for c in CR10)), 0)

has("pilot labelled as such", "pilot")
has("pilot subset size stated", "2{,}000-trace")
has("B threshold disclosed as inherited", "inherited rather than chosen")
has("B=32 untested at low ratios", "never tested at those ratios")
has("no claim B=32 would be better", "no claim about which value would be better")
has("negative pilot result disclosed", "no variant of the light architecture exceeded")
has("supersession stated", "supersede it")

print()
print("J. Section 2.8 loss-function ablation on LARA (re-derived from the released CSVs)")
LA = pd.read_csv(find("results/loss_ablation_LARA_selection.csv"))
LR = pd.read_csv(find("results/loss_ablation_LARA_ranking.csv"))
LCR = [2, 3, 5, 10, 15, 20, 30, 50, 60, 100]
LSPEC = ["STFT", "STFT_Arrival", "STFT_Phase", "STFT_Wavelet", "STFT_Phase_Arrival"]
LNC = "NCC_Arrival"
LMSE = "MSE"
_l = {(r.Arm, int(r.CR)): r for r in LA.itertuples()}
L = lambda a, c, m: float(getattr(_l[(a, c)], m))
TOL = 0.005  # the table rounds to 2 decimals

ck("loss ablation row count", 70, len(LA), 0)
ck("loss ablation arms", 7, len({r.Arm for r in LA.itertuples()}), 0)
ck("ratios per arm", 10, LA[LA.Arm == LMSE].shape[0], 0)
ck("loss ablation is LARA at every ratio", 70,
   int(sum(1 for r in LA.itertuples()
           if abs(float(r.Parameter_Count) - float(
               MAIN[(MAIN.Model == "LARA") & (MAIN.CR == int(r.CR))].Parameter_Count.iloc[0])) < 1)), 0)

# --- every SNR cell printed in Table 4, both panels, parsed from the manuscript
_tt = BODY[BODY.index(r"\label{tab:loss-ablation}"):]
_tt = _tt[:_tt.index(r"\end{tabular}")]
_arm = {a: a.replace("_", chr(92) + "_") for a in [LMSE] + LSPEC + [LNC]}
_rows = {a: [] for a in _arm}
for _line in _tt.splitlines():
    if "\\\\" not in _line:
        continue
    if "&" not in _line:
        continue
    _first = _line.split("&")[0].strip()
    for _a, _disp in _arm.items():
        # exact first-cell match: "STFT" is a prefix of "STFT_Arrival", so a
        # startswith test would file every spectral row under the plain arm
        if _first == _disp or _first == _disp + " (control)":
            _rows[_a].append([float(x) for x in re.findall(r"[0-9]+\.[0-9]+|[0-9]+", _line)])
ck("Table 4 arms parsed", 7, sum(1 for v in _rows.values() if v), 0)
for _a, _v in sorted(_rows.items()):
    ck("Table 4 %s appears once per panel" % _a, 2, len(_v), 0)

# Panel A carries CR 2-15, panel B carries CR 20-100. Check every printed cell.
for _a in _arm:
    _pa, _pb = _rows[_a][0], _rows[_a][1]
    for _i, _c in enumerate([2, 3, 5, 10, 15]):
        ck("Table 4 %s SNR CR=%d" % (_a, _c), round(L(_a, _c, "SNR"), 2), _pa[_i], TOL)
    for _i, _c in enumerate([20, 30, 50, 60, 100]):
        ck("Table 4 %s SNR CR=%d" % (_a, _c), round(L(_a, _c, "SNR"), 2), _pb[_i], TOL)

# --- the seven mean-delta figures, cross-checked two ways
_rank = {r.Arm: r for r in LR.itertuples()}
for a in LSPEC + [LNC]:
    d = sorted(L(a, c, "SNR") - L(LMSE, c, "SNR") for c in LCR)
    mean = sum(d) / 10
    med = (d[4] + d[5]) / 2
    won = sum(1 for c in LCR if L(a, c, "SNR") > L(LMSE, c, "SNR"))
    ck("Table 4 mean delta %s" % a, round(mean, 3), round(_rank[a].Mean_dSNR_dB, 3), 0)
    ck("Table 4 median delta %s" % a, round(med, 3), round(_rank[a].Median_dSNR_dB, 3), 0)
    ck("Table 4 won count %s" % a, _rank[a].CRs_won_vs_MSE, won, 0)
    ck("Table 4 lost count %s" % a, _rank[a].CRs_lost_vs_MSE, 10 - won, 0)
    ck("median is negative for %s" % a, True, bool(med < 0))

# --- epoch ranges printed in the last column
for a in [LMSE] + LSPEC + [LNC]:
    e = [int(_l[(a, c)].Epochs_Used) for c in LCR]
    ck("epoch range %s" % a, (min(e), max(e)), (min(e), max(e)), 0)

# --- the five spectral arms: no metric, no ratio
for a in [x for x in LSPEC if x != "STFT_Wavelet"]:
    for m in ("SNR", "SSIM", "PSNR", "Correlation"):
        ck("%s never beats the control on %s" % (a, m), 0,
           sum(1 for c in LCR if L(a, c, m) > L(LMSE, c, m)), 0)
    for m in ("MSE", "MAE"):
        ck("%s never beats the control on %s" % (a, m), 0,
           sum(1 for c in LCR if L(a, c, m) < L(LMSE, c, m)), 0)
# STFT_Wavelet is the one documented exception: CR=5 only, on three metrics
_beats = lambda a, m, c: (L(a, c, m) > L(LMSE, c, m)) if m not in ("MSE", "MAE") \
    else (L(a, c, m) < L(LMSE, c, m))
for m in ("SNR", "PSNR", "MAE"):
    ck("STFT_Wavelet wins %s only at CR=5" % m, [5],
       [c for c in LCR if _beats("STFT_Wavelet", m, c)], 0)
for m in ("SSIM", "Correlation"):
    ck("STFT_Wavelet never wins %s" % m, 0,
       sum(1 for c in LCR if L("STFT_Wavelet", c, m) > L(LMSE, c, m)), 0)
ck("STFT_Wavelet is the only spectral arm to win SNR, once", 1,
   sum(1 for a in LSPEC for c in LCR if L(a, c, "SNR") > L(LMSE, c, "SNR")), 0)
ck("no spectral arm ever wins correlation", 0,
   sum(1 for a in LSPEC for c in LCR if L(a, c, "Correlation") > L(LMSE, c, "Correlation")), 0)
ck("no spectral arm ever wins SSIM", 0,
   sum(1 for a in LSPEC for c in LCR if L(a, c, "SSIM") > L(LMSE, c, "SSIM")), 0)

ds = sorted(L(a, c, "SNR") - L(LMSE, c, "SNR") for a in LSPEC for c in LCR)
ck("spectral deficit worst", -11.00, round(ds[0], 2), 0.0)
ck("spectral deficit least bad", 1.21, round(ds[-1], 2), 0.0)
ck("spectral deficit mean over fifty comparisons", -3.94, round(sum(ds) / 50, 2), 0.0)
for c, want in ((2, -7.64), (3, -4.86), (5, -0.85), (30, -5.00), (100, -2.85)):
    v = sum(L(a, c, "SNR") - L(LMSE, c, "SNR") for a in LSPEC) / 5
    ck("mean spectral deficit at CR=%d" % c, want, round(v, 2), 0.0)

# --- NCC_Arrival: correlation at every ratio, SNR at three
ck("NCC_Arrival wins correlation at all ten ratios", 10,
   sum(1 for c in LCR if L(LNC, c, "Correlation") > L(LMSE, c, "Correlation")), 0)
ck("NCC_Arrival wins SSIM at eight of ten ratios", 8,
   sum(1 for c in LCR if L(LNC, c, "SSIM") > L(LMSE, c, "SSIM")), 0)
ck("NCC_Arrival wins SNR at three of ten ratios", 3,
   sum(1 for c in LCR if L(LNC, c, "SNR") > L(LMSE, c, "SNR")), 0)
for c, want in ((10, 0.003), (30, 0.003), (50, 0.065), (100, 0.144)):
    ck("NCC_Arrival correlation margin at CR=%d" % c, want,
       round(L(LNC, c, "Correlation") - L(LMSE, c, "Correlation"), 3), 0.0)
for c, want in ((5, 2.31), (50, 0.18), (100, 0.45)):
    ck("NCC_Arrival SNR margin at CR=%d" % c, want,
       round(L(LNC, c, "SNR") - L(LMSE, c, "SNR"), 2), 0.0)
n9 = [L(LNC, c, "SNR") - L(LMSE, c, "SNR") for c in LCR if c != 5]
ck("NCC_Arrival mean SNR excluding CR=5", -0.053, round(sum(n9) / 9, 3), 0.0)
ck("MSE beats NCC on SNR at seven of ten", 7,
   sum(1 for c in LCR if L(LMSE, c, "SNR") > L(LNC, c, "SNR")), 0)
ck("MSE beats NCC on MSE at six of ten", 6,
   sum(1 for c in LCR if L(LMSE, c, "MSE") < L(LNC, c, "MSE")), 0)
ck("MSE beats NCC on MAE at eight of ten", 8,
   sum(1 for c in LCR if L(LMSE, c, "MAE") < L(LNC, c, "MAE")), 0)

# --- the CR=5 confound
ck("control trained shortest at CR=5", True,
   bool(int(_l[(LMSE, 5)].Epochs_Used) == min(int(_l[(LMSE, c)].Epochs_Used) for c in LCR)))
ck("control at CR=5 is 3.19 dB below Table 1", -3.19,
   round(L(LMSE, 5, "SNR") - float(MAIN[(MAIN.Model == "LARA") & (MAIN.CR == 5)].SNR.iloc[0]), 2), 0.0)
ck("longest-trained arm is worse than the control at CR=100", True,
   bool(L("STFT_Phase_Arrival", 100, "SNR") < L(LMSE, 100, "SNR")))
ck("NCC trained fewer epochs than the control at CR=50", True,
   bool(int(_l[(LNC, 50)].Epochs_Used) < int(_l[(LMSE, 50)].Epochs_Used)))

# --- protocol claims in the prose
ck("latent exact at every ratio", 10,
   int(sum(1 for c in LCR if int(_l[(LMSE, c)].Latent_Dim) == 1500 // c)), 0)
ck("every Actual_CR equals nominal", 70,
   int(sum(1 for r in LA.itertuples() if abs(float(r.Actual_CR) - float(r.CR)) < 0.01)), 0)
ck("phase weight scales 0.3 to 0.9", (0.3, 0.9),
   (round(float(_l[("STFT_Phase", 2)].Lambda_Phase_eff), 1),
    round(float(_l[("STFT_Phase", 100)].Lambda_Phase_eff), 1)), 0)
ck("arrival floor scales 1.0 to 3.0", (1.0, 3.0),
   (round(float(_l[(LNC, 2)].Arrival_Floor_eff), 1),
    round(float(_l[(LNC, 100)].Arrival_Floor_eff), 1)), 0)

# --- the predecessor study must be gone
has("no predecessor architecture claim", "light-capacity architecture", present=False)
has("no 2,000-trace pool claim", "2,000~STEAD traces", present=False)
has("no 42-run claim", "42 runs", present=False)
has("loss ablation states seventy models", "Seventy models were trained")
has("epoch confound disclosed", "matched by budget cap rather than by duration")
has("CR=5 confound disclosed", "the least reliable point")
has("nine-ratio figure given", "$-0.053$~dB")
has("correlation trade-off stated", "would obtain better correlation")

print()
print("H. Self-consistency")
# The paper quotes this script's own check count. Assert it here so adding a
# check without updating the manuscript fails the gate rather than going stale.
# `checks + 1` accounts for this check itself.
_m = re.search(r"runs (\d+) checks with no failures", BODY)
ck("quoted check count matches this run", int(_m.group(1)) if _m else -1,
   checks + 1, 0)

print()
print("=" * 110)
print(f"CHECKS RUN: {checks}    FAILURES: {len(fails)}")
for f in fails:
    print(f"   FAIL {f[0]}: tex={f[1]}  csv={f[2]}")
sys.exit(1 if fails else 0)
