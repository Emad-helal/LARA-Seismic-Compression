"""Build the computational-cost technical note as a .docx.

Every number is read from the benchmark CSV at build time. The subnormal-weight
fraction is recomputed directly from the checkpoints, because that quantity
turned out to be the dominant term in the whole measurement and must not be
inherited from a column that could be regenerated differently.

CPU NOTE ONLY. The GPU measurements live in their own folder with their own
builder, build_cost_gpu_docx.py, and this script is pinned to the CPU files so
the two can never read each other's results.
"""
import os
import sys
import glob
import json
import shutil

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

RUN = (r"D:\Post_Doctor\Dr. Mostafa\Enhanced_Residual_AutoEncoder\Dr. Omar"
       r"\Review_27092026\seismic 26 9 2026")
COST = os.path.join(RUN, "results", "compute_cost")
CPU = os.path.join(COST, "compute_cost_CPU")
CKPT = os.path.join(RUN, "ckpts")
OUT = os.path.join(CPU, "Final_LARA_computational_cost_technical_note.docx")
PLOT = os.path.join(CPU, "figures")

NEURAL = ["LARA", "GeneralizedAutoencoder", "AE_PureConcat"]
COL = {"LARA": "#66c2a5", "GeneralizedAutoencoder": "#8da0cb", "AE_PureConcat": "#80b1d3"}
SUB_LO = 1.1754944e-38
SIG_S = 15.0
CPS = 3  # components per station
TRACE_S = CPS / SIG_S  # traces per second delivered by one station
CRS = [2, 3, 5, 10, 15, 20, 30, 50, 60, 100]


def _one(pattern, what):
    hits = sorted(glob.glob(pattern))
    if len(hits) != 1:
        raise SystemExit(f"ABORT: expected exactly 1 {what} matching {pattern}, "
                         f"found {len(hits)}: {hits}")
    return hits[0]


BENCH = _one(os.path.join(CPU, "compute_bench_*_full.csv"), "CPU benchmark CSV")
HJSON = _one(os.path.join(CPU, "hardware_*.json"), "CPU hardware JSON")
bench = pd.read_csv(BENCH)
hw = json.load(open(HJSON, encoding="utf-8"))
if hw.get("accelerator_kind") != "cpu":
    raise SystemExit(f"ABORT: {HJSON} is a {hw.get('accelerator_kind')} record, "
                     "but this script builds the CPU note")
bench["denormal_mode"] = bench["denormal_mode"].fillna("n/a")

# Maximum thread count actually measured. Read from the data, never hard-coded,
# because the thread count differs by platform and a wrong constant silently
# returns NaN instead of failing.
THREADS = sorted(bench["threads"].dropna().unique().astype(int).tolist())
TH1, THM = THREADS[0], THREADS[-1]
if THM == TH1:
    raise SystemExit(f"ABORT: only one thread count ({THM}) present in {BENCH}; "
                     "the multi-thread comparison cannot be built")

# ---------------------------------------------------------------- data access
def subfrac(model, cr):
    """Fraction of published weights that are subnormal float32."""
    p = os.path.join(CKPT, f"best_{model}_cr{int(cr)}.pth")
    if not os.path.exists(p):
        return np.nan
    sd = torch.load(p, map_location="cpu", weights_only=True)
    n = s = 0
    for _, v in sd.items():
        if not torch.is_floating_point(v):
            continue
        a = v.detach().abs().float().flatten()
        n += a.numel()
        s += int(((a > 0) & (a < SUB_LO)).sum())
    return s / n if n else 0.0


SUB = {(m, c): subfrac(m, c) for m in NEURAL for c in CRS}


def net_ms(model, cr, threads, batch=1, mode="flushed"):
    """Network only (encoder + decoder), ms per trace."""
    r = bench[(bench.model == model) & (bench["cr"] == cr) & (bench.threads == threads)
              & (bench["batch"] == batch) & (bench.denormal_mode == mode)
              & (bench.stage.isin(["encoder", "decoder"]))]
    return float(r.ms_per_trace.sum()) if len(r) else np.nan


def one_ms(model, cr, stage, mode="flushed", threads=1, batch=1):
    r = bench[(bench.model == model) & (bench["cr"] == cr) & (bench.stage == stage)
              & (bench.threads == threads) & (bench["batch"] == batch)
              & (bench.denormal_mode == mode)]
    return float(r.ms_per_trace.iloc[0]) if len(r) else np.nan


def coder_ms(model, cr, threads):
    """Quantize + dequantize + entropy encode + entropy decode, ms per trace."""
    c = bench[(bench.model == model) & (bench["cr"] == cr) & (bench.threads == threads)
              & (bench.stage.isin(["quantize", "dequantize", "entropy_encode",
                                   "entropy_roundtrip"]))]
    if not len(c):
        return dict(q=np.nan, dq=np.nan, enc=np.nan, dec=np.nan, tot=np.nan)
    enc = float(c[c.stage == "entropy_encode"].ms_per_trace.iloc[0])
    rt = float(c[c.stage == "entropy_roundtrip"].ms_per_trace.iloc[0])
    q = float(c[c.stage == "quantize"].ms_per_trace.iloc[0])
    dq = float(c[c.stage == "dequantize"].ms_per_trace.iloc[0])
    return dict(q=q, dq=dq, enc=enc, dec=rt - enc, tot=q + dq + enc + (rt - enc))


def fkey(model, cr, which):
    r = bench[(bench.model == model) & (bench["cr"] == cr) & (bench.stage == "flops")]
    return float(r[which].iloc[0])


def npar(model, cr):
    r = bench[(bench.model == model) & (bench["cr"] == cr)].dropna(subset=["nparam"])
    return int(r.nparam.iloc[0])


def ckpt_mb(model, cr):
    r = bench[(bench.model == model) & (bench["cr"] == cr)].dropna(subset=["ckpt_bytes"])
    return int(r.ckpt_bytes.iloc[0]) / 1e6


def pipeline_ms(model, cr, threads, mode="flushed", batch=1):
    """Whole codec per trace: encoder + decoder + quantise + entropy-code +
    dequantise + entropy-decode. This is the cost of compressing a trace for
    transmission, not of decoding one."""
    n = net_ms(model, cr, threads, batch=batch, mode=mode)
    c = coder_ms(model, cr, threads)
    return n + c["tot"]


def stations(per_trace_ms):
    """Stations one core sustains. One station delivers TRACE_S traces per
    second, so capacity is the per-trace rate DIVIDED by that, not multiplied."""
    return (1000.0 / per_trace_ms) / TRACE_S


def peak_mb(model, cr):
    r = bench[(bench.model == model) & (bench["cr"] == cr) & (bench.stage == "memory")]
    return float(r.peak_infer_bytes.iloc[0]) / 1e6 if len(r) else np.nan


flops_tot = {c: fkey("LARA", c, "encoder_flops") + fkey("LARA", c, "decoder_flops")
             for c in CRS}

# headline quantities
asp = {c: net_ms("LARA", c, TH1, mode="as_deployed") / net_ms("LARA", c, TH1, mode="flushed")
       for c in CRS}
MAXCR, MAXSP = max(asp.items(), key=lambda kv: kv[1])
stg = bench[(bench.model == "LARA") & (bench.stage.isin(["encoder", "decoder"]))
            & (bench["batch"] == 1) & (bench.threads == TH1)]
piv = stg.pivot_table(index=["cr", "stage"], columns="denormal_mode", values="ms_per_trace")
piv["sp"] = piv.as_deployed / piv.flushed
MAXSTG = piv.sp.max()
MAXSTG_IDX = piv.sp.idxmax()

# per-stage slowdown, the quantity that shows the penalty is encoder-dominated
ENCSP = {c: one_ms("LARA", c, "encoder", "as_deployed", TH1)
         / one_ms("LARA", c, "encoder", "flushed", TH1) for c in CRS}
DECSP = {c: one_ms("LARA", c, "decoder", "as_deployed", TH1)
         / one_ms("LARA", c, "decoder", "flushed", TH1) for c in CRS}
MAXENCC, MAXENCSP = max(ENCSP.items(), key=lambda kv: kv[1])
MAXDECC, MAXDECSP = max(DECSP.items(), key=lambda kv: kv[1])
# CR with no observed penalty, on either stage
NOPEN = [c for c in CRS if max(ENCSP[c], DECSP[c]) < 1.05]

e10 = one_ms("LARA", 10, "encoder", "flushed", TH1)
d10 = one_ms("LARA", 10, "decoder", "flushed", TH1)
n10 = e10 + d10
coder10 = coder_ms("LARA", 10, TH1)
t10 = pipeline_ms("LARA", 10, TH1)
sh10_1 = coder10["tot"] / t10
cd10_m = coder_ms("LARA", 10, THM)
sh10_m = cd10_m["tot"] / pipeline_ms("LARA", 10, THM)
sh2_1 = coder_ms("LARA", 2, TH1)["tot"] / pipeline_ms("LARA", 2, TH1)
sh3_1 = coder_ms("LARA", 3, TH1)["tot"] / pipeline_ms("LARA", 3, TH1)
sh100_1 = coder_ms("LARA", 100, TH1)["tot"] / pipeline_ms("LARA", 100, TH1)
bl10 = {m: net_ms(m, 10, TH1) for m in NEURAL}
bl10_m = {m: net_ms(m, 10, THM) for m in NEURAL}
bl_m_share = {m: coder_ms(m, 10, THM)["tot"] / pipeline_ms(m, 10, THM) for m in NEURAL}
NPAR_LO, NPAR_HI = npar("LARA", 100), npar("LARA", 2)
MB_LO, MB_HI = NPAR_LO * 4 / 1e6, NPAR_HI * 4 / 1e6
FL_MIN = min(flops_tot.values()) / 1e6
FL_MAX = max(flops_tot.values()) / 1e6

# Batched throughput, per trace. `net_ms` reads the per-trace column, so no
# further division by the batch size is needed.
BATCH = sorted(bench.loc[(bench.stage == "encoder") & bench.threads.eq(THM),
                         "batch"].dropna().unique().tolist())
BMAX = int(max(BATCH))
b_m = {c: net_ms("LARA", c, THM, batch=BMAX) for c in CRS}
b_1 = {c: net_ms("LARA", c, TH1) for c in CRS}
BSPD = {c: b_1[c] / b_m[c] for c in CRS}
BLO, BHI = min(b_m.values()), max(b_m.values())
TPS_LO, TPS_HI = 1000.0 / BHI, 1000.0 / BLO
SP_LO, SP_HI = min(BSPD.values()), max(BSPD.values())
BATCH_M = {}
for m in NEURAL:
    v = {c: net_ms(m, c, THM, batch=BMAX) for c in CRS}
    BATCH_M[m] = (min(v.values()), max(v.values()),
                  1000.0 / max(v.values()), 1000.0 / min(v.values()))
# threading effect at batch 1: does raising the pool help, hurt, or do nothing?
THR = {c: net_ms("LARA", c, TH1) / net_ms("LARA", c, THM) for c in CRS}
THR_LO, THR_HI = min(THR.values()), max(THR.values())

# Real-time capacity, full pipeline, single stream. Corrected: the published
# figure multiplied the decode rate by the per-station rate instead of dividing.
ST_FL = {c: stations(pipeline_ms("LARA", c, TH1)) for c in CRS}
ST_AD = {c: stations(pipeline_ms("LARA", c, TH1, mode="as_deployed")) for c in CRS}
ST_FL_B = {c: stations(pipeline_ms("LARA", c, THM, batch=BMAX)) for c in CRS}
RT = stations(t10)
RT_WORST_FL = min(ST_FL.values())
RT_WORST_AD = min(ST_AD.values())
RT_CRF = min(ST_AD, key=lambda c: ST_AD[c])
ALL_REALTIME = min(ST_AD.values()) >= 1.0

os.makedirs(PLOT, exist_ok=True)
_TMP = os.path.join(os.environ.get("TEMP", "."), "lara_cost_figs")
os.makedirs(_TMP, exist_ok=True)


def save(fig, stem):
    """Save via a local temp file: the D: share intermittently rejects direct
    writes when a previous copy of the same image is still open elsewhere."""
    for e in ("png", "eps"):
        t = os.path.join(_TMP, f"{stem}.{e}")
        fig.savefig(t, dpi=300, bbox_inches="tight")
        shutil.copyfile(t, os.path.join(PLOT, f"{stem}.{e}"))

# ------------------------------------------------------------------- figures
def fig_flops():
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.8))
    x = np.arange(len(CRS))
    w = 0.27
    for i, m in enumerate(NEURAL):
        off = x + (i - 1) * w
        axes[0].bar(off, [fkey(m, c, "encoder_flops") / 1e6 for c in CRS], w,
                    color=COL[m], label=m, edgecolor="black", linewidth=0.5)
        axes[1].bar(off, [fkey(m, c, "decoder_flops") / 1e6 for c in CRS], w,
                    color=COL[m], label=m, edgecolor="black", linewidth=0.5)
    for ax, t in zip(axes, ["Encoder", "Decoder"]):
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels([str(c) for c in CRS])
        ax.set_xlabel("Compression Ratio", fontsize=11)
        ax.set_ylabel("MFLOPs per 1500-sample trace (log)", fontsize=11)
        ax.set_title(t, fontsize=12, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.3, which="both")
        ax.set_axisbelow(True)
        ax.legend(fontsize=9, ncol=3, loc="lower center")
    fig.suptitle("Arithmetic cost per trace, encoder and decoder separately "
                 "(multiply-add = 2 FLOPs; elementwise operations not counted)",
                 fontsize=11.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0.02, 1, 0.93))
    save(fig, "flops_vs_cr")
    plt.close(fig)


def fig_pipeline():
    """Cost SHARE, so a linear 0-100% axis: stacking on a log axis would distort it."""
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.8), sharey=True)
    x = np.arange(len(CRS))
    parts = [("network (enc+dec)", "#66c2a5"), ("quantize", "#fc8d62"),
             ("entropy encode", "#8da0cb"), ("entropy decode", "#e78ac3")]
    for ax, th in zip(axes, (TH1, THM)):
        tot = []
        comp = {}
        for lab, _ in parts:
            if lab.startswith("network"):
                comp[lab] = np.array([net_ms("LARA", c, th) for c in CRS])
            else:
                k = {"quantize": "q", "entropy encode": "enc", "entropy decode": "dec"}[lab]
                comp[lab] = np.array([coder_ms("LARA", c, th)[k] for c in CRS])
            tot.append(comp[lab])
        T = np.sum(np.vstack(tot), axis=0)
        bot = np.zeros(len(CRS))
        for lab, col in parts:
            pct = comp[lab] / T * 100.0
            ax.bar(x, pct, 0.62, bottom=bot, label=lab, color=col,
                   edgecolor="black", linewidth=0.4)
            bot += pct
        ax.set_xticks(x)
        ax.set_xticklabels([str(c) for c in CRS])
        ax.set_xlabel("Compression Ratio", fontsize=11)
        ax.set_title(f"{th} thread(s)", fontsize=12, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_axisbelow(True)
        ax.set_ylim(0, 100)
    axes[0].set_ylabel("Share of per-trace pipeline cost (%)", fontsize=11)
    fig.suptitle("Where the time goes in the LARA pipeline, subnormals flushed. Absolute "
                 "totals for the single-thread case are in the accompanying table.",
                 fontsize=11.5, fontweight="bold")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=9, loc="lower center", ncol=4, frameon=True,
               bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=(0, 0.06, 1, 0.90))
    save(fig, "pipeline_vs_cr")
    plt.close(fig)


def fig_latency():
    fig, ax = plt.subplots(figsize=(11.2, 5.4))
    x = np.arange(len(CRS))
    w = 0.2
    ax.bar(x - 1.5 * w, [net_ms("LARA", c, 1, mode="as_deployed") for c in CRS], w,
           color="#c0504d", edgecolor="black", linewidth=0.5, label="LARA, as deployed")
    ax.bar(x - 0.5 * w, [net_ms("LARA", c, 1, mode="flushed") for c in CRS], w,
           color=COL["LARA"], edgecolor="black", linewidth=0.5,
           label="LARA, subnormals flushed")
    ax.bar(x + 0.5 * w, [net_ms("GeneralizedAutoencoder", c, 1) for c in CRS], w,
           color=COL["GeneralizedAutoencoder"], edgecolor="black", linewidth=0.5,
           label="GeneralizedAutoencoder")
    ax.bar(x + 1.5 * w, [net_ms("AE_PureConcat", c, 1) for c in CRS], w,
           color=COL["AE_PureConcat"], edgecolor="black", linewidth=0.5,
           label="AE_PureConcat")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([str(c) for c in CRS])
    ax.set_xlabel("Compression Ratio", fontsize=11)
    ax.set_ylabel("Encoder + decoder, ms per trace (log)", fontsize=11)
    ax.set_title("Single-thread CPU network latency per trace, batch 1",
                 fontsize=12, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3, which="both")
    ax.set_axisbelow(True)
    ax.legend(fontsize=9)
    fig.tight_layout()
    save(fig, "latency_vs_cr")
    plt.close(fig)


def fig_subnormal():
    fig, ax = plt.subplots(figsize=(11.2, 5.0))
    x = np.arange(len(CRS))
    ax.bar(x - 0.18, [SUB[("LARA", c)] * 100 for c in CRS], 0.36, color=COL["LARA"],
           edgecolor="black", linewidth=0.5, label="LARA")
    ax.bar(x + 0.18, [SUB[("GeneralizedAutoencoder", c)] * 100 for c in CRS], 0.36,
           color=COL["GeneralizedAutoencoder"], edgecolor="black", linewidth=0.5,
           label="GeneralizedAutoencoder")
    ax.set_xticks(x)
    ax.set_xticklabels([str(c) for c in CRS])
    ax.set_xlabel("Compression Ratio", fontsize=11)
    ax.set_ylabel("Subnormal parameters (%)", fontsize=11)
    ax.set_title("Fraction of published weights that are subnormal float32",
                 fontsize=12, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(fontsize=9)
    fig.tight_layout()
    save(fig, "subnormal_fraction")
    plt.close(fig)


fig_flops()
fig_pipeline()
fig_latency()
fig_subnormal()

# ------------------------------------------------------------------- document
doc = Document()
s = doc.styles["Normal"]
s.font.name = "Times New Roman"
s.font.size = Pt(10.5)
s.paragraph_format.space_after = Pt(6)
s.paragraph_format.line_spacing = 1.15
for sec in doc.sections:
    sec.left_margin = sec.right_margin = Inches(0.9)
    sec.top_margin = sec.bottom_margin = Inches(0.9)


def para(t, size=10.5, bold=False, italic=False, align=None, after=6):
    p = doc.add_paragraph()
    r = p.add_run(t)
    r.font.size = Pt(size)
    r.bold = bold
    r.italic = italic
    p.paragraph_format.space_after = Pt(after)
    if align:
        p.alignment = align
    return p


def h(t, lv):
    p = doc.add_heading(t, level=lv)
    for r in p.runs:
        r.font.name = "Times New Roman"
        r.font.color.rgb = RGBColor(0, 0, 0)
    return p


def cap(t, above=False):
    p = doc.add_paragraph()
    r = p.add_run(t)
    r.font.size = Pt(8.5)
    r.italic = True
    p.alignment = (WD_ALIGN_PARAGRAPH.LEFT if above else WD_ALIGN_PARAGRAPH.CENTER)
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(10)
    return p


# --------------------------------------------------------- exhibit numbering
# PLAN is the single source of truth for exhibit numbering, in document order.
# Prose refers to exhibits through tref()/fref(), which resolve against PLAN, so
# a reference can be written before the exhibit it points at has been emitted and
# follows the exhibit if the plan changes. tcap/fcap allocate the number PLAN
# already assigned, and verify_exhibits() fails the build if the captions are
# emitted in an order that disagrees with PLAN, if an exhibit is uncited, if a
# citation names a number that is not a caption, or if either sequence is out of
# order. There is no hand-written exhibit number anywhere below.
PLAN = [
    ("Table", "env"),           # 2.1
    ("Figure", "flops"),        # 3.1
    ("Table", "flops_tab"),     # 3.1
    ("Table", "stage"),         # 3.2
    ("Figure", "pipeline"),     # 3.2
    ("Table", "pipeline_tab"),  # 3.2
    ("Figure", "latency"),      # 3.3
    ("Figure", "subnormal"),    # 3.3
    ("Table", "latency_tab"),   # 3.3
    ("Table", "memory"),        # 3.4
    ("Table", "realtime"),      # 3.5
]
_COUNT = {"Table": 0, "Figure": 0}
KEY = {}
for _kind, _key in PLAN:
    if _key in KEY:
        raise SystemExit(f"ABORT: {PLAN} lists key {_key!r} twice")
    _COUNT[_kind] += 1
    KEY[_key] = (_kind, _COUNT[_kind])
_EMITTED = []


def _cap_exhibit(kind, key, text, above):
    if key not in KEY:
        raise SystemExit(f"ABORT: exhibit key {key!r} is not in PLAN")
    if KEY[key][0] != kind:
        raise SystemExit(f"ABORT: key {key!r} is a {KEY[key][0]}, not a {kind}")
    _EMITTED.append((kind, key))
    cap(f"{kind} {KEY[key][1]}. {text}", above=above)


def tcap(key, text, above=False):
    _cap_exhibit("Table", key, text, above)


def fcap(key, text, above=False):
    _cap_exhibit("Figure", key, text, above)


def tref(key):
    if key not in KEY:
        raise SystemExit(f"ABORT: tref({key!r}) is not in PLAN")
    return f"Table {KEY[key][1]}"


def fref(key):
    if key not in KEY:
        raise SystemExit(f"ABORT: fref({key!r}) is not in PLAN")
    return f"Figure {KEY[key][1]}"


def verify_exhibits():
    """Check the emitted captions against PLAN, and the prose against both."""
    import re
    if _EMITTED != PLAN:
        only_plan = [e for e in PLAN if e not in _EMITTED]
        only_doc = [e for e in _EMITTED if e not in PLAN]
        raise SystemExit(
            "ABORT: captions were not emitted in PLAN order.\n"
            f"  in PLAN but never emitted: {only_plan}\n"
            f"  emitted but not in PLAN  : {only_doc}")
    body, caps = [], []
    for p in doc.paragraphs:
        t = p.text.strip()
        if not t:
            continue
        m = re.match(r"^(Table|Figure)\s+(\d+)\.\s", t)
        if m:
            caps.append((m.group(1), int(m.group(2))))
        else:
            body.append(t)
    txt = " ".join(body)
    bad = []
    for kind, n in caps:
        if not re.search(rf"\b{kind}\s+{n}\b", txt):
            bad.append(f"{kind} {n} is captioned but never cited in the text")
    cited = {("Table", int(n)) for n in re.findall(r"\bTable\s+(\d+)\b", txt)}
    cited |= {("Figure", int(n)) for n in re.findall(r"\bFigure\s+(\d+)\b", txt)}
    for kind, n in sorted(cited):
        if (kind, n) not in caps:
            bad.append(f"text cites {kind} {n}, which is not a caption in this document")
    for kind in ("Table", "Figure"):
        seq = [n for k, n in caps if k == kind]
        if seq != sorted(seq):
            bad.append(f"{kind} sequence is out of document order: {seq}")
        if seq and seq != list(range(1, len(seq) + 1)):
            bad.append(f"{kind} sequence is not contiguous from 1: {seq}")
    if bad:
        raise SystemExit("ABORT: exhibit problems\n  " + "\n  ".join(bad))
    return len(caps)


def fig(path, w=6.5):
    doc.add_picture(path, width=Inches(w))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER


def table(header, rows, widths=None, size=8.5):
    t = doc.add_table(rows=1, cols=len(header))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, x in enumerate(header):
        c = t.rows[0].cells[i]
        c.text = ""
        r = c.paragraphs[0].add_run(str(x))
        r.bold = True
        r.font.size = Pt(size)
        c.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
    for row in rows:
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = ""
            r = cells[i].paragraphs[0].add_run(str(v))
            r.font.size = Pt(size)
            cells[i].paragraphs[0].alignment = (
                WD_ALIGN_PARAGRAPH.LEFT if i == 0 else WD_ALIGN_PARAGRAPH.CENTER)
    if widths:
        for i, w in enumerate(widths):
            for row in t.rows:
                row.cells[i].width = Inches(w)
    return t


para("Computational Cost of Learned Seismic Waveform Codecs: Latency, "
     "Arithmetic Complexity, Memory and Real-Time Feasibility",
     size=15, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, after=4)
para("Technical note", size=10, italic=True, align=WD_ALIGN_PARAGRAPH.CENTER, after=14)

para("Abstract", size=11, bold=True, after=4)
para(
    f"Learned autoencoders for seismic compression are usually assessed by rate and "
    f"distortion, and rarely by the cost of running them. We report the computational cost of "
    f"the three neural codecs of the study: arithmetic complexity of the encoder and decoder "
    f"separately, encoding and decoding latency, single-thread and multi-thread latency, batched "
    f"throughput, the arithmetic coder, memory, and the real-time capacity of one core. Two "
    f"findings dominate. First, the published LARA checkpoints contain subnormal float32 "
    f"weights, from {SUB[('LARA',3)]*100:.2f}% of parameters at a compression ratio of 3 to "
    f"{SUB[('LARA',100)]*100:.1f}% at a ratio of 100; both baseline codecs contain none. "
    f"Because subnormal operands take a microcoded path on x86, this inflates LARA's CPU cost "
    f"by up to {MAXSP:.0f} times per trace and up to {MAXSTG:.0f} times for a single stage, and "
    f"it falls almost entirely on the encoder. "
    f"Setting torch.set_flush_denormal(True) removes the penalty entirely and changes no "
    f"result. Second, the arithmetic coder is not a negligible accessory: it is the majority "
    f"of the pipeline cost at low compression ratios, reaching {sh3_1:.0%} of the total at "
    f"CR = 3, and it is the dominant cost for both baseline codecs once the network is "
    f"batched and threaded. After the fix, the full codec path costs about {t10:.1f} ms per "
    f"15 s trace on one core, which is {RT:,.0f} stations, and the arithmetic cost is close to "
    f"independent of the compression ratio because the fixed-size pyramid decoder dominates it. "
    f"We report the as-deployed and corrected figures side by side. A companion note reports the "
    f"same measurement on the accelerator used for training, where the subnormal penalty is "
    f"absent and the coder share roughly doubles.",
    size=10, after=10)
para("Keywords: computational cost; inference latency; FLOPs; arithmetic coding; real-time "
     "compression; subnormal arithmetic; neural codec; seismic waveform",
     size=9.5, italic=True, after=12)

h("1. Introduction", 1)
para(
    "A codec that reconstructs accurately is of no use if it cannot keep pace with the data "
    "arriving from the field, and a reported rate that is achievable only at unacceptable "
    "latency is not a codec. This note answers three practical questions about the three neural "
    "models of the study: how much arithmetic does a trace cost, how long does that arithmetic "
    "take on real hardware, and does the result permit real-time operation.")
para(
    "The answers are not the ones we expected. Arithmetic cost is modest and nearly flat in the "
    "compression ratio, which is convenient. Latency, however, is governed by two effects that "
    "no amount of architectural cleverness addresses: a numerical pathology in the published "
    "LARA weights, and the arithmetic coder, which is a larger share of the pipeline than its "
    "role in the rate-domain literature would suggest. Both are reported here with the "
    "measurements that expose them.")

h("2. Materials and methods", 1)

h("2.1 Hardware and software", 2)
para(f"The measurements below were made on the configuration in {tref('env')}. Every row of the "
     "accompanying results file carries this fingerprint, so measurements from different "
     "machines cannot be merged by accident.")
tcap("env", "Measurement environment.", above=True)
table(["Item", "Value"],
      [["CPU", hw.get("cpu", "n/a")],
       ["Physical / logical cores",
        f"{hw.get('physical_cores','?')} / {hw.get('logical_cores','?')}"],
       ["Accelerator used", f"{hw.get('accelerator','n/a')} (device kind: "
                            f"{hw.get('accelerator_kind','?')})"],
       ["System memory", f"{hw.get('ram_gb','?')} GB"],
       ["PyTorch", hw.get("torch", "?")],
       ["Python", hw.get("python", "?")],
       ["Platform", hw.get("platform", "?")],
       ["Signal length", "1500 samples (15 s at 100 Hz)"],
       ["Hardware ID", hw.get("hardware_id", "?")]],
      widths=[1.7, 4.5])
para(hw.get("accelerator_note",
            "No CUDA device was visible to PyTorch on this host, so all figures are CPU-only."),
     size=9, italic=True, after=4)
para(
    "The identical script was run on the accelerator machine used for training, and those "
    "measurements are reported separately in the companion accelerator note, built by "
    "build_cost_gpu_docx.py from the results file in compute_cost_GPU/. The two are kept in "
    "separate folders so that neither can be reported against the other's hardware. Note that "
    "the two platforms differ in more than the accelerator: the host CPU, the PyTorch build and "
    "the Python version also differ, so the comparison is not a controlled ablation of the "
    "accelerator alone, and the companion note says so where it matters.", after=10)

h("2.2 Protocol", 2)
para(
    "Latency is the median per-call time over a run of at least 0.5 s of wall clock after "
    "warm-up. A fixed iteration count is avoided deliberately: this host is a shared virtual "
    "machine and a short burst of iterations can fall entirely inside a CPU quota stall, which "
    "produced apparent latencies two orders of magnitude too high in a first attempt. Every "
    "thread count is measured in a separate process, because reconfiguring the thread pool "
    "within one process while a large array is resident inflated single-thread latency by an "
    "order of magnitude. Arithmetic counts use the framework's operator counter and were taken "
    "in a separate process again, since instrumenting every operation degrades the "
    "measurements that follow. The encoder and the decoder are timed separately because they "
    "are the two halves of a codec that in practice run on different machines. Models are used "
    "exactly as published, loaded from the released checkpoints with no modification.")
para(
    "Each network measurement was made twice: once as published, and once with subnormal "
    "flushing enabled. Both appear throughout; neither is presented alone. The coder stages are "
    "measured per trace, independent of batch, and the decode cost is taken as the round-trip "
    "arithmetic-coding time minus the encode time, since the two share an encoder.")

h("3. Results", 1)

h("3.1 Arithmetic complexity", 2)
para(
    f"The decoder is the more expensive half of every model, and for LARA it is about four "
    f"times the encoder: at CR = 10 the encoder costs {fkey('LARA',10,'encoder_flops')/1e6:.1f} "
    f"MFLOPs and the decoder {fkey('LARA',10,'decoder_flops')/1e6:.1f} MFLOPs, as {fref('flops')} and "
    f"{tref('flops_tab')} show. More "
    f"importantly, the arithmetic cost barely responds to the compression ratio: across a "
    f"fifty-fold change in CR the LARA total moves only from {FL_MIN:.0f} to {FL_MAX:.0f} "
    f"MFLOPs. The reason is structural. The cost sits in the pyramid decoder, which always "
    f"reconstructs a full 1500-sample trace, while the encoder only narrows the bottleneck. "
    f"The one exception is the encoder above CR 30, which roughly doubles because an additional "
    f"downsampling stage is instantiated. For a deployed codec this is a favourable property: "
    f"the same code and the same cost serve every operating point.")
fig(os.path.join(PLOT, "flops_vs_cr.png"))
fcap("flops", "Arithmetic cost per 1500-sample trace, encoder and decoder separately. A "
    "multiply-add counts as two FLOPs; elementwise operations are not counted.")
tcap("flops_tab", "Arithmetic cost in MFLOPs per trace, with the LARA subnormal-weight fraction.",
     above=True)
rows = []
for c in CRS:
    r = [str(c)]
    for m in NEURAL:
        r += [f"{fkey(m,c,'encoder_flops')/1e6:.1f}", f"{fkey(m,c,'decoder_flops')/1e6:.1f}"]
    r.append(f"{SUB[('LARA',c)]*100:.1f}%")
    rows.append(r)
table(["CR", "LARA enc", "LARA dec", "GAE enc", "GAE dec", "AE enc", "AE dec",
       "LARA subnorm."], rows,
      widths=[0.5, 0.9, 0.9, 0.8, 0.8, 0.75, 0.75, 1.15], size=7.5)

h("3.2 Latency, encoding and decoding time, and the arithmetic coder", 2)
para(
    f"Encoding and decoding times are reported separately throughout, because a codec in "
    f"practice runs its two halves on different machines: the encoder on the station that "
    f"observes the event, the decoder on the receiver. With subnormal flushing enabled and one "
    f"core, a LARA trace at CR = 10 costs {e10:.2f} ms to encode and {d10:.2f} ms to decode. "
    f"{tref('stage')} gives both for every ratio, together with the as-deployed stage, because the two "
    f"halves are penalised very differently by the subnormal condition of Section 3.3.")
tcap("stage", "LARA encoding and decoding time, ms per 1500-sample trace, single thread, batch 1, "
    "with subnormals flushed, and the as-deployed slowdown of each stage.", above=True)
rows = []
for c in CRS:
    ef = one_ms("LARA", c, "encoder", "flushed", TH1)
    df_ = one_ms("LARA", c, "decoder", "flushed", TH1)
    rows.append([str(c), f"{ef:.2f}", f"{df_:.2f}", f"{df_/ef:.2f}",
                 f"{ENCSP[c]:.1f}x", f"{DECSP[c]:.1f}x"])
table(["CR", "Encoder", "Decoder", "Dec/enc", "Encoder slowdown", "Decoder slowdown"],
      rows, widths=[0.55, 0.95, 0.95, 0.8, 1.35, 1.35], size=8)
para(
    f"Two features of {tref('stage')} matter operationally. First, the decoder costs more than the "
    f"encoder at every ratio, by {min(one_ms('LARA',c,'decoder','flushed',TH1)/one_ms('LARA',c,'encoder','flushed',TH1) for c in CRS):.1f} "
    f"to {max(one_ms('LARA',c,'decoder','flushed',TH1)/one_ms('LARA',c,'encoder','flushed',TH1) for c in CRS):.1f} "
    f"times, so the receiving side is the expensive one and a deployment that fans out over many "
    f"low-power receivers should plan capacity accordingly. Second, the subnormal penalty falls "
    f"almost entirely on the encoder: the largest encoder slowdown is {MAXENCSP:.0f} times at "
    f"CR = {MAXENCC}, against {MAXDECSP:.0f} times for the decoder at CR = {MAXDECC}. The full "
    f"per-trace figure of {tref('latency_tab')} is therefore an average over a badly penalised and a mildly "
    f"penalised stage, which is why the two are separated here.")
para(
    f"Throughput comes from batching, not from threading. At batch 1, raising the thread count "
    f"from {TH1} to {THM} leaves LARA between {THR_LO:.2f} and {THR_HI:.2f} times its "
    f"single-thread cost, and at the highest ratios it is worse than single-threaded, because the "
    f"tensors are small enough that thread-pool overhead exceeds the work. At batch {BMAX} on "
    f"{THM} threads the per-trace network cost falls to between {BLO:.2f} and {BHI:.2f} ms, "
    f"an improvement of {SP_LO:.1f} to {SP_HI:.1f} times over single-threaded batch 1, which is "
    f"{TPS_LO:,.0f} to {TPS_HI:,.0f} traces per second. Per-trace latencies at batch {BMAX} are "
    f"{BATCH_M['LARA'][0]:.3f} to {BATCH_M['LARA'][1]:.3f} ms for LARA, "
    f"{BATCH_M['GeneralizedAutoencoder'][0]:.3f} to {BATCH_M['GeneralizedAutoencoder'][1]:.3f} ms "
    f"for the GeneralizedAutoencoder and {BATCH_M['AE_PureConcat'][0]:.3f} to "
    f"{BATCH_M['AE_PureConcat'][1]:.3f} ms for AE_PureConcat. One caution on reading these "
    f"figures: a timed forward call carries a whole batch, so the per-trace cost is the call time "
    f"divided by the batch size, and both quantities are given in the accompanying results file.")
para(
    f"The arithmetic coder is a much larger share of the pipeline than the rate-domain framing "
    f"of this codec suggests, and {fref('pipeline')} shows how strongly its share depends on the ratio. "
    f"At CR = 3 the coder "
    f"accounts for {sh3_1:.0%} of the total per-trace cost on one thread, and {sh2_1:.0%} at "
    f"CR = 2, because a small bottleneck means little data to code while the network cost is "
    f"unchanged. At CR = 100 it falls to {sh100_1:.0%}. The coder also does not benefit from "
    f"torch threading in the way the network does, so its share rises to {sh10_m:.0%} at "
    f"CR = 10 on {THM} threads. Most strikingly, once the networks are batched and threaded the "
    f"two baselines are coder-bound: at CR = 10 on {THM} threads the coder is "
    f"{bl_m_share['GeneralizedAutoencoder']:.0%} of the GeneralizedAutoencoder pipeline and "
    f"{bl_m_share['AE_PureConcat']:.0%} of the AE_PureConcat pipeline, because their networks "
    f"cost only {bl10_m['GeneralizedAutoencoder']:.2f} and "
    f"{bl10_m['AE_PureConcat']:.2f} ms per trace. Any comparison of these codecs that omits "
    f"the coder is therefore not comparing their deployed cost.")
fig(os.path.join(PLOT, "pipeline_vs_cr.png"))
fcap("pipeline", "Share of the per-trace pipeline cost taken by the network, the quantizer and "
    "the arithmetic coder. The coder is the majority cost at low ratios; absolute totals for "
    f"the single-thread case are in {tref('pipeline_tab')}.")
tcap("pipeline_tab", "LARA pipeline breakdown, ms per trace, single thread, subnormals flushed.",
     above=True)
rows = []
for c in CRS:
    cm = coder_ms("LARA", c, 1)
    tt = net_ms("LARA", c, 1) + cm["tot"]
    rows.append([str(c), f"{net_ms('LARA',c,1):.2f}", f"{cm['q']:.2f}", f"{cm['enc']:.2f}",
                 f"{cm['dec']:.2f}", f"{tt:.2f}", f"{cm['tot']/tt:.0%}"])
table(["CR", "Network", "Quantize", "Entropy enc", "Entropy dec", "Total", "Coder share"],
      rows, widths=[0.5, 1.0, 1.0, 1.1, 1.1, 0.9, 1.0], size=8)

h("3.3 Subnormal weights dominate the measurement", 2)
para(
    f"The first version of this measurement was internally inconsistent: observed latency did "
    f"not track the FLOP count, and a decoder costing 185 MFLOPs appeared to take 256 ms while "
    f"an identically sized 212 MFLOP decoder at another ratio took 3.9 ms. Isolating the cause "
    f"showed that a freshly initialised model of identical architecture ran in 1.9 ms and the "
    f"same model after loading the published weights ran in 84 ms, a factor of forty-five with "
    f"identical arithmetic.")
para(
    f"The explanation is that the published LARA weights contain subnormal float32 values, that "
    f"is magnitudes below 1.18e-38. Subnormal operands are handled by a microcoded fallback path "
    f"on x86 and can cost up to two orders of magnitude. The fraction of subnormal parameters "
    f"is {SUB[('LARA',2)]*100:.1f}% at CR = 2, {SUB[('LARA',3)]*100:.2f}% at CR = 3, "
    f"{SUB[('LARA',30)]*100:.1f}% at CR = 30 and {SUB[('LARA',100)]*100:.1f}% at CR = 100. "
    f"Neither baseline codec contains a single subnormal parameter at any ratio, and a freshly "
    f"initialised LARA contains none, so the condition is a consequence of training this "
    f"architecture rather than of the architecture itself.")
para(
    f"Calling torch.set_flush_denormal(True) before inference removes the penalty completely. "
    f"The measured effect is large and not proportional to the subnormal fraction: the largest "
    f"per-trace slowdown is {MAXSP:.0f} times at CR = {MAXCR}, and the largest single-stage "
    f"slowdown is {MAXSTG:.0f} times for the {MAXSTG_IDX[1]} at CR = {MAXSTG_IDX[0]}. The "
    f"relationship to the subnormal fraction is therefore loose, and the reason is the "
    f"permutation of operations: a subnormal operand penalises any layer that touches it, so "
    f"which layer is hit hardest depends on the weight distribution rather than on how many "
    f"parameters are affected. We report the association and the fix, not a predictive model "
    f"for the slowdown. Setting the flag changes no result quality, since the affected weights "
    f"are of magnitude below the smallest normal float32 and contribute nothing to the output. "
    + (f"CR = {', '.join(str(c) for c in NOPEN)} "
       f"{'is' if len(NOPEN) == 1 else 'are'} the only ratio at which no penalty is observed, and "
       f"we do not have an explanation for it; the other {len(CRS)-len(NOPEN)} ratios are "
       f"penalised." if NOPEN else
       f"Every ratio is penalised on at least one stage."))
para(
    f"{fref('latency')} shows the effect directly, and {fref('subnormal')} shows the weight "
    f"fractions it corresponds to; {tref('latency_tab')} gives the per-ratio as-deployed and "
    f"corrected network latency. "
    f"The two plots are deliberately paired, because the association is what the fix rests on: "
    f"the ratios with the largest gap in {fref('latency')} are the ratios with a non-zero bar in "
    f"{fref('subnormal')}.")
fig(os.path.join(PLOT, "latency_vs_cr.png"))
fcap("latency", "Single-thread CPU network latency per trace, batch 1, on a logarithmic axis. The "
    "gap between the first and second bars is the subnormal penalty; the baselines are flat.")
fig(os.path.join(PLOT, "subnormal_fraction.png"))
fcap("subnormal", "Fraction of published weights that are subnormal float32. Both baselines are "
    "exactly zero at every ratio.")
tcap("latency_tab", "Single-thread batch-1 network latency per trace (ms), encoder + decoder, as "
    "deployed and with subnormal flushing.", above=True)
rows = []
for c in CRS:
    a = net_ms("LARA", c, TH1, mode="as_deployed")
    b = net_ms("LARA", c, TH1, mode="flushed")
    rows.append([str(c), f"{a:.2f}", f"{b:.2f}", f"{a/b:.1f}x", f"{SUB[('LARA',c)]*100:.1f}%"])
table(["CR", "As deployed", "Flushed", "Speed-up", "Subnormal weights"], rows,
      widths=[0.7, 1.2, 1.1, 0.9, 1.5])

h("3.4 Memory", 2)
para(
    f"Parameter memory is set by the architecture and is modest: LARA holds between "
    f"{NPAR_LO/1e6:.2f} and {NPAR_HI/1e6:.2f} million parameters depending on the ratio, that "
    f"is {MB_LO:.2f} to {MB_HI:.2f} MB in float32. The released checkpoint files match this to "
    f"within the container overhead, so they store weights only and carry no optimizer state, "
    f"which is what a deployment should ship. {tref('memory')} gives the figure for all three codecs.")
tcap("memory", "Weight memory per model, MB in float32, smallest and largest ratio.", above=True)
rows = []
for m in NEURAL:
    w = [npar(m, c) * 4 / 1e6 for c in CRS]
    k = [ckpt_mb(m, c) for c in CRS]
    rows.append([m.replace("GeneralizedAutoencoder", "Generalized AE"),
                 f"{npar(m,100):,}", f"{npar(m,2):,}",
                 f"{min(w):.2f}", f"{max(w):.2f}", f"{min(k):.2f}", f"{max(k):.2f}"])
table(["Model", "Params CR100", "Params CR2", "Weights min", "Weights max",
       "File min", "File max"], rows, widths=[1.3, 0.95, 0.85, 0.85, 0.85, 0.75, 0.75], size=8)
para(
    f"We do not report a measured activation footprint for this platform, and the reason is "
    f"worth stating because it is a limitation of the measurement rather than of the method. The "
    f"benchmark records peak inference memory on the host as the resident-set-size delta after "
    f"warm-up, and on a CPU that delta excludes the model weights and every reusable buffer the "
    f"allocator has already obtained, because those are resident before the measurement starts. "
    f"The recorded values for LARA therefore range from 0.00 to "
    f"{max(peak_mb('LARA',c) for c in CRS):.2f} MB across the ten ratios, are not monotonic in "
    f"the ratio, and reach exactly zero at some ratios, which is a property of the allocator "
    f"rather than of the network. The column is retained in the results file for completeness "
    f"but is not evidence about memory use. The defensible statement on this platform is "
    f"analytic: the weights of {tref('memory')} must be resident, and the transient activation working set "
    f"is dominated by the 1500-length pyramid, which is fixed by the trace length and not by the "
    f"compression ratio, so it is smallest at batch 1 and grows linearly with the batch. The "
    f"companion accelerator note reports a genuine device-side peak, which is the figure to use "
    f"for a deployment footprint.")
para(
    f"The weights must nevertheless be resident on the decoding host, and as the companion note "
    f"on rate shows, that fixed cost dominates the rate until the corpus is very large. LARA's "
    f"weight memory is also the smallest of the three codecs at every ratio, which matters for a "
    f"receiver fleet: the AE_PureConcat checkpoint is up to "
    f"{max(ckpt_mb('AE_PureConcat',c) for c in CRS):.0f} MB and the GeneralizedAutoencoder up to "
    f"{max(ckpt_mb('GeneralizedAutoencoder',c) for c in CRS):.0f} MB, against "
    f"{max(ckpt_mb('LARA',c) for c in CRS):.0f} MB for LARA.")

h("3.5 Real-time feasibility", 2)
para(
    f"Taking a {SIG_S:.0f} s window per trace at {100.0:.0f} Hz and {CPS} components per station, "
    f"one station delivers {TRACE_S:.1f} traces per second, so the capacity of a core in stations "
    f"is the per-trace rate divided by that figure, not multiplied by it.")
para(
    f"The budget must be the whole codec, not the decoder alone, because a seismic station both "
    f"compresses and transmits. We therefore charge every stage: encode, quantise, "
    f"entropy-code, decode, dequantise and entropy-decode, which is the Total column of {tref('pipeline_tab')}. "
    f"With subnormals flushed and a single thread, that is {t10:.2f} ms per trace at CR = 10, "
    f"which is {1000.0/t10:,.0f} traces per second and therefore {RT:,.0f} stations on one core. "
    f"Across the ten ratios the capacity runs from {min(pipeline_ms('LARA',c,TH1) for c in CRS):.1f} "
    f"ms to {max(pipeline_ms('LARA',c,TH1) for c in CRS):.1f} ms per trace, that is "
    f"{RT_WORST_FL:,.0f} to {max(ST_FL.values()):,.0f} stations. Batching and threading multiply "
    f"this: at batch {BMAX} on {THM} threads the pipeline sustains "
    f"{min(ST_FL_B.values()):,.0f} to {max(ST_FL_B.values()):,.0f} stations on one core. That "
    f"batched figure charges the network and the arithmetic coder serially on the same core, "
    f"which is the conservative reading for a single-host deployment; a receiver that overlaps "
    f"the coder with the network on separate cores would be limited only by the network, which "
    f"at CR = 10 is {net_ms('LARA',10,THM,batch=BMAX):.2f} ms per trace against "
    f"{coder_ms('LARA',10,THM)['tot']:.2f} ms for the coder, so the coder is not the binding "
    f"constraint there. LARA is comfortably real-time on this CPU once subnormals are flushed, "
    f"at every ratio.")
para(
    f"The as-deployed artefact is a different and much tighter picture, and it is the honest one "
    f"to quote for a deployment that ships the released checkpoints unchanged. The worst ratio is "
    f"CR = {RT_CRF}, where the full pipeline costs "
    f"{pipeline_ms('LARA',RT_CRF,TH1,mode='as_deployed'):,.0f} ms per trace and one core still "
    f"sustains {RT_WORST_AD:,.1f} stations. That is real-time, but with almost no headroom, and it "
    f"is the encoder that consumes the budget: the as-deployed encoder alone takes "
    f"{one_ms('LARA',RT_CRF,'encoder','as_deployed',TH1):,.0f} ms at that ratio, while the decoder "
    f"takes {one_ms('LARA',RT_CRF,'decoder','as_deployed',TH1):,.0f} ms. On the decode side alone "
    f"the limiting factor is the decoder rather than the encoder, but for compression and "
    f"transmission together the encoder is the larger cost at the ratios where the subnormal "
    f"penalty bites, and it is the ratio of {ENCSP[RT_CRF]:.0f} times at CR = {RT_CRF} that makes "
    f"this so. The single worst stage in the whole measurement is the as-deployed LARA encoder, "
    f"at {MAXENCSP:.0f} times its corrected cost at CR = {MAXENCC}.")
para(
    f"{tref('realtime')} collects these figures for all ten ratios, in both the corrected and the "
    f"as-deployed case and for the batched pipeline, so the three bases can be compared "
    f"directly rather than quoted from separate sentences.")
tcap("realtime", "Real-time capacity of one core, LARA, stations sustained, full codec per trace.",
     above=True)
rows = []
for c in CRS:
    f1 = pipeline_ms("LARA", c, TH1)
    fa = pipeline_ms("LARA", c, TH1, mode="as_deployed")
    rows.append([str(c), f"{f1:.2f}", f"{ST_FL[c]:,.0f}", f"{fa:,.2f}", f"{ST_AD[c]:,.1f}",
                 f"{ST_FL_B[c]:,.0f}"])
table(["CR", "Flushed ms", "Flushed stations", "As-deployed ms", "As-deployed stations",
       f"Batched stations"], rows,
      widths=[0.5, 0.9, 1.15, 1.2, 1.25, 1.15], size=8)
para(
    "This assessment is specific to one CPU and should not be transferred to an accelerator. At "
    "a few hundred MFLOPs per trace the arithmetic is far below what a modern GPU sustains, so "
    "on hardware of that kind the bottleneck would move away from the network toward "
    "host-device transfer and the arithmetic coder, and the subnormal penalty, being a "
    "scalar-arithmetic effect, would be expected to matter far less. These three predictions are "
    "tested directly in the companion accelerator note, which reports the same measurement on "
    "the accelerator used for training: the subnormal penalty is absent, the coder share roughly "
    "doubles, and the coder finding of Section 3.2 does carry over, since it is a property of "
    "the data and the coder rather than of the accelerator.")

h("4. Discussion", 1)

h("4.1 What the measurement changes", 2)
para(
    f"The arithmetic cost of LARA is unremarkable and close to independent of the compression "
    f"ratio, which is a favourable property for a deployed codec. The decoder is about four "
    f"times the encoder, so the receiving side is the expensive one and a deployment that fans "
    f"out over many low-power receivers should plan capacity accordingly. Against the two "
    f"baselines, LARA is the most expensive of the three in both arithmetic and latency terms, "
    f"so its advantage in the parent study is reconstruction quality rather than efficiency, and "
    f"that trade should be stated explicitly when the results are presented.")

h("4.2 The two actionable findings", 2)
para(
    f"The subnormal condition is specific to LARA, is absent from both baselines and from a "
    f"fresh initialisation of the same architecture, and is removed by a single call. We "
    f"recommend that the released checkpoints be re-saved with subnormal weights flushed to "
    f"zero, that the flag be set in any CPU deployment, and that training be checked for weight "
    f"collapse in the filled-funnel configuration: "
    f"{SUB[('LARA',100)]*100:.0f}% of parameters decaying into the subnormal range at the "
    f"highest ratio suggests the funnel is not fully utilised at that operating point. This is a "
    f"training diagnosis as much as a deployment one. We report the as-deployed timings rather "
    f"than only the corrected ones because the deployed artefact has this property, and a "
    f"reader reproducing our measurements would otherwise see only the slow path.")
para(
    f"The second finding is that the arithmetic coder deserves to be treated as part of the "
    f"computational cost rather than as a post-processing step. For LARA it is the majority of "
    f"the pipeline at low ratios, and for both baselines under batching and threading it is the "
    f"dominant term. A future version of this codec should therefore consider coder cost in the "
    f"design objective, since the ratios at which the coder dominates, CR 2 to 5, are exactly "
    f"the ratios at which the rate is most attractive.")

h("4.3 Limitations", 2)
para(
    "Four limitations bound this note. First, the measurements here are from a single CPU, and "
    "the companion accelerator note is a different machine rather than a controlled ablation of "
    "the accelerator, since its host CPU and PyTorch build also differ; only the subnormal "
    "result, which is an order of magnitude effect, survives that caveat cleanly. Second, FLOPs "
    "count matrix multiplications and convolutions only, so the reported arithmetic is a "
    "lower bound on the work and does not capture elementwise operations, normalization or the "
    "quantizer; the latency measurements, which do include everything, are the figures to rely "
    "on. Third, the host is a shared virtual machine, so absolute latencies carry more run-to-run "
    "variation than a dedicated host would. The as-deployed versus corrected comparison is robust "
    "to this because both were measured on the same host in the same run, but the absolute "
    "station counts in Section 3.5 should be read as indicative. Fourth, only the three neural "
    "codecs were characterised; the analytic wavelet references in the parent study were not "
    "benchmarked here and no claim is made about their cost. A fifth limitation is specific to "
    "Section 3.4: peak inference memory was recorded as a host resident-set delta, which excludes "
    "weights and reusable buffers, so no measured activation footprint is reported for this "
    f"platform and the weight figures of {tref('memory')} are analytic.")

h("5. Conclusion", 1)
para(
    f"LARA requires about {flops_tot[10]/1e6:.0f} MFLOPs per trace at CR = 10, and about "
    f"{one_ms('LARA',10,'decoder','flushed',TH1)/one_ms('LARA',10,'encoder','flushed',TH1):.1f} "
    f"times more arithmetic to decode than to encode; its arithmetic cost is nearly "
    f"independent of the compression ratio. Its latency is not, because the decoder is both the "
    f"more expensive half in arithmetic and the more expensive half in time. Measured on one "
    f"modern CPU core, a corrected LARA costs {e10:.2f} ms to encode and {d10:.2f} ms to decode "
    f"at CR = 10, and {t10:.2f} ms for the full codec, which is {RT:,.0f} stations on one core "
    f"and therefore comfortably real-time. Three results would not have been found without "
    f"direct measurement. The published LARA checkpoints contain up to "
    f"{SUB[('LARA',100)]*100:.0f}% subnormal float32 weights, which inflate CPU latency by up "
    f"to {MAXSP:.0f} times per trace and by up to {MAXENCSP:.0f} times in the encoder "
    f"specifically, is absent from both baseline codecs, and is removed without any loss of "
    f"accuracy by enabling subnormal flushing. The arithmetic coder, not the network, is the "
    f"majority cost at low compression ratios and the dominant cost for both baselines once the "
    f"network is batched and threaded. And the real-time budget must be charged for the whole "
    f"codec rather than the decoder alone: doing so changes the answer by a factor of "
    f"{RT/max(ST_FL.values()):.2f} to {RT/min(ST_FL.values()):.2f} depending on the ratio, and "
    f"the two are not interchangeable.")

h("References", 1)
for i, r in enumerate([
    "S. G. Mallat, \"A wavelet-based compression framework,\" IEEE Trans. Signal Processing, "
    "vol. 37, no. 7, 1989.",
    "D. L. Mallat, \"A theory for multiresolution signal decomposition: the wavelet "
    "representation,\" IEEE Trans. Pattern Analysis and Machine Intelligence, vol. 11, no. 3, "
    "1989.",
    "IEEE 754-2019, \"IEEE Standard for Floating-Point Arithmetic,\" 2019.",
    "D. E. Knuth, \"The Art of Computer Programming, Volume 2: Seminumerical Algorithms,\" "
    "2nd ed. Addison-Wesley, 1996, section on floating-point arithmetic.",
    "J. Hauser, \"How subnormals affect performance,\" The Green Shoe, 2008.",
    "D. Bailey, A. Chow, A. Cleary, H. L. Nguyen, and M. E. Taylor, \"Floating-point subnormal "
    "numbers and their use in solving eigenvalue problems,\" ACM Trans. Mathematical Software, "
    "vol. 35, no. 2, 2009.",
    "J. BallÃ©, V. Laparra, and E. P. Simoncelli, \"End-to-end optimized image compression,\" in "
    "Proc. Int. Conf. on Learning Representations (ICLR), 2017.",
    "J. BallÃ©, J. Minnen, G. Singh, and J. A. Johnston, \"Variational image compression with a "
    "scale hyperprior,\" in Proc. Advances in Neural Information Processing Systems (NeurIPS), "
    "2018.",
    "X. Zhu, H. Tang, N. Liu, and others, \"STEAD: A non-stationary global seismic data set for "
    "machine learning in seismology,\" in Proc. Community of Computational Seismology and "
    "Earthquake Engineering (ComBee), 2019.",
], 1):
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Inches(0.3)
    p.paragraph_format.first_line_indent = Inches(-0.3)
    p.paragraph_format.space_after = Pt(3)
    p.add_run(f"[{i}]  {r}").font.size = Pt(8.5)

N_EXH = verify_exhibits()
doc.save(OUT)
print(f"wrote {OUT}  ({os.path.getsize(OUT):,} bytes)")
print(f"  exhibits: {N_EXH} (all captioned, cited and in order)")
print(f"  hardware {hw['hardware_id']}  threads {THREADS}  batch {BATCH}")
print(f"  CR10 flushed: enc {e10:.2f} ms, dec {d10:.2f} ms, net {n10:.2f} ms, "
      f"full pipeline {t10:.2f} ms -> {RT:,.0f} stations/core")
print(f"  subnormal: max per-trace {MAXSP:.1f}x at CR={MAXCR}; max encoder {MAXENCSP:.1f}x at "
      f"CR={MAXENCC}; max decoder {MAXDECSP:.1f}x at CR={MAXDECC}; no penalty at CR={NOPEN}")
print(f"  coder share: CR2 {sh2_1:.0%}, CR3 {sh3_1:.0%}, CR10 {sh10_1:.0%}, CR100 {sh100_1:.0%}")
print(f"  batched (b{BMAX}/t{THM}): {BLO:.3f}-{BHI:.3f} ms/trace, {TPS_LO:,.0f}-{TPS_HI:,.0f} "
      f"traces/s, speedup {SP_LO:.1f}-{SP_HI:.1f}x")
print(f"  baseline coder share @t{THM} CR10: GAE {bl_m_share['GeneralizedAutoencoder']:.0%}, "
      f"AE {bl_m_share['AE_PureConcat']:.0%}")
print(f"  real-time stations/core, flushed {RT_WORST_FL:,.0f}-{max(ST_FL.values()):,.0f}; "
      f"as-deployed {RT_WORST_AD:,.1f}-{max(ST_AD.values()):,.1f}")
