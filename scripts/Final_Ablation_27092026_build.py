"""Build Final_Ablation_27092026.ipynb from the clean 20092026 source.

Cells that must not change (dataset, shared blocks, HARA arms, plotting) are
extracted VERBATIM from the source notebook.  class LARA is extracted VERBATIM
from Final_LARA_27092026.ipynb so the topology identity is guaranteed by
construction, not by retyping.
"""
import json
import os
import re
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

SRC_ABL = r"C:\Users\master\Final_HARA\Final_Ablation_20092026.ipynb"
SRC_LARA = r"C:\Users\master\Final_HARA\Seismic 26092025\Final_LARA_27092026.ipynb"
BUILD = os.path.dirname(os.path.abspath(__file__))
OUT = (r"D:\Post_Doctor\Dr. Mostafa\Enhanced_Residual_Autoencoder\Dr. Omar"
       r"\Review_27092026\Final_Ablation_27092026.ipynb")

# ---------------------------------------------------------------- extract
src = json.load(open(SRC_ABL, encoding="utf-8"))
A = {i: "".join(c["source"]) for i, c in enumerate(src["cells"])}
S = {i: A[i] for i in A if src["cells"][i]["cell_type"] == "code"}
report = []


def read(p):
    return open(os.path.join(BUILD, p), encoding="utf-8").read().rstrip() + "\n"


def patch(text, old, new, n, label):
    """Replace exactly n occurrences or fail loudly."""
    got = text.count(old)
    assert got == n, f"{label}: expected {n} occurrence(s) of {old!r}, found {got}"
    report.append(f"  patched {label:<34} x{got}")
    return text.replace(old, new)


# ---- trainer: source cell 10, pinned to the CONFIG block -------------------
tr = S[10]
tr = patch(tr, "    if max_epochs is None:\n        max_epochs = 120\n",
           "    if max_epochs is None:\n        max_epochs = MAX_EPOCHS\n",
           1, "trainer: max_epochs -> MAX_EPOCHS")
lr_chain = """    # EXACT ABLATION STUDY LEARNING RATES - CORRECTED TO MATCH ORIGINAL ABLATION
    if cr <= 3:
        lr = 0.0002
    elif cr <= 5:
        lr = 0.0002
    elif cr <= 15:
        lr = 0.0002
    else:
        lr = 0.0002
"""
assert lr_chain in tr, "trainer: lr chain not found verbatim"
tr = patch(tr, lr_chain, "    lr = LR\n", 1, "trainer: lr -> LR")
tr = patch(tr, "        weight_decay=1e-5\n", "        weight_decay=WEIGHT_DECAY\n",
           1, "trainer: weight_decay -> WEIGHT_DECAY")
tr = patch(tr, "        patience=5,\n        min_lr=1e-6\n",
           "        patience=SCHED_PATIENCE,\n        min_lr=MIN_LR\n",
           1, "trainer: scheduler patience/min_lr")
tr = patch(tr, "    patience = 8\n", "    patience = PATIENCE\n", 1, "trainer: patience -> PATIENCE")
tr = patch(tr, "clip_grad_norm_(model.parameters(), 1.0)",
           "clip_grad_norm_(model.parameters(), GRAD_CLIP)", 1, "trainer: grad clip -> GRAD_CLIP")
tr = patch(tr, "f'best_{model_name}_cr{cr}.pth'", "best_file(model_name, cr)",
           3, "trainer: checkpoint path -> CKPT_DIR")

# ---- metric helpers: kept VERBATIM from the source cell 12, which defined
#      calculate_psnr / calculate_ssim_1d / calculate_model_size before
#      evaluate_model. Only evaluate_model itself is replaced.
src_eval = S[12]
split_at = src_eval.index("def evaluate_model(")
metric_helpers = src_eval[:split_at].rstrip()
for fn in ("def calculate_psnr(", "def calculate_ssim_1d(", "def calculate_model_size("):
    assert fn in metric_helpers, f"metric helper {fn} not found in source cell 12"
report.append("  kept verbatim from src cell 12: calculate_psnr / calculate_ssim_1d "
              "/ calculate_model_size")

n15 = metric_helpers + "\n\n\n" + read("n15_evaluate.py")

# ---- class LARA, verbatim ---------------------------------------------------
lara_nb = json.load(open(SRC_LARA, encoding="utf-8"))
lara_cells = ["".join(c["source"]) for c in lara_nb["cells"] if c["cell_type"] == "code"]
lara_src = next(s for s in lara_cells if "class LARA(nn.Module)" in s)
i0 = lara_src.index("class LARA(nn.Module)")
i1 = lara_src.find("\nclass ", i0 + 10)
lara_class = (lara_src[i0:i1] if i1 > 0 else lara_src[i0:]).rstrip()
assert lara_class.startswith("class LARA(nn.Module):"), "LARA extraction lost the class header"
assert lara_class.rstrip().endswith('print("[OK] Fallback architecture created")'), \
    f"LARA extraction looks truncated, ends with: {lara_class[-80:]!r}"

n10 = lara_class + """


# =============================
# The five ablation arms
# =============================
# Light_Capacity_LARA IS the LARA model implemented and trained in
# Final_LARA_27092026 -- class LARA above, used as-is with
# variant="pyramid_funnel32".  The two are not separate arms: the figure and
# table legends read Light_Capacity_LARA everywhere.
#
# Light_Capacity_HARA is still defined above for provenance but is
# deliberately NOT registered: the old light arm used T=min(latent, L), a
# linear-up ResBlock decoder and a single head, which is a different topology
# from LARA.  Comparing both would mix "light-capacity" with "LARA" and
# duplicate the comparison already made by the original study.
LARA_VARIANT = "pyramid_funnel32"


def Light_Capacity_LARA(cr):
    \"\"\"The LARA arm, pinned to the variant used in Final_LARA_27092026.\"\"\"
    return LARA(cr, variant=LARA_VARIANT)


ABLATION_MODELS = [
    ("Base_Residual_Attention", Base_Residual_Attention),
    ("No_Attention_HARA",        No_Attention_HARA),
    ("Plain_Conv_HARA",          Plain_Conv_HARA),
    ("Light_Capacity_LARA",      Light_Capacity_LARA),
    ("Heavy_Capacity_HARA",      Heavy_Capacity_HARA),
]
"""

# ---------------------------------------------------------------- markdown
def md(t):
    return {"cell_type": "markdown", "metadata": {}, "source": t.splitlines(True)}


MD = {}
MD[0] = md("""# HARA Component Ablation Study - LARA-Aligned (27092026)

Re-runs the original five-arm component ablation so that it is directly
comparable with the final LARA study. Everything about the data, the component,
the filter and the training budget has been aligned; the arms now read
`Light_Capacity_LARA` instead of `Light_Capacity_HARA`.

## Arms

| Arm | What it isolates |
|---|---|
| `Base_Residual_Attention` | full HARA: global attention + 4 ResBlocks + multi-scale loss |
| `No_Attention_HARA` | removes the attention block |
| `Plain_Conv_HARA` | removes the residual blocks (attention kept) |
| `Light_Capacity_LARA` | LARA (`variant="pyramid_funnel32"`) - the final proposed model |
| `Heavy_Capacity_HARA` | doubles stem and base width |

## What changed against Final_Ablation_20092026

| Setting | 20092026 | 27092026 |
|---|---|---|
| Data root | `D:\\HARA_Final\\merged.csv` / `.hdf5` | `D:\\STEAD\\merged.csv` / `.hdf5` |
| Magnitude filter | `> 3.0` | `> 2.5` (pool 25,091) |
| Component | 0 | **2** (a different physical component) |
| `N_EVAL` | 20 | **200** |
| SNR | pooled over the flattened test set | per-sample, then averaged |
| Light arm | `Light_Capacity_HARA` | `Light_Capacity_LARA` |
| Checkpoints | working directory | `<run>/ckpts/` |
| CR check | none | 50-combination pre-flight + post-run report |

The training recipe, the compression ratios, the architecture of the four HARA
arms and the figure code are unchanged.

## Scope

This study compares **LARA as a whole** against the HARA arms. LARA differs from
them along four axes at once - filled funnel, B=32 stem, six-level pyramid
decoder, and a two-layer refinement head - so it does **not** isolate which
individual component produces the gain. A component-level attribution would
need LARA ablations (funnel off, linear decoder, single head) as extra arms.
""")

MD[2] = md("""## Data

Aligned with `Final_LARA_27092026`: the STEAD merge, `magnitude > 2.5`,
`distance_km <= 60`, **component 2**, and the same 80/10/10 split with
`SEED = 42`. Roughly 25,091 traces qualify.

Component 0 and component 2 are different physical channels on this build
(measured mean correlation -0.07, 0 of 60 traces above 0.90), which is why the
component had to match the LARA study rather than the original ablation.
""")

MD[7] = md("""## HARA backbone and the four HARA arms

Verbatim from the original study so the comparison is not confounded by an
accidental edit. `Light_Capacity_HARA` is still defined - the notebook stays
self-contained and reproducible - but it is **not** in `ABLATION_MODELS`
(see the next cell): `Light_Capacity_LARA` replaces it.
""")

MD[9] = md("""## LARA - the light-capacity arm

`class LARA` below is copied **verbatim** from `Final_LARA_27092026.ipynb`
(sha256 of the class text matches the executed GPU-run copy), so
`Light_Capacity_LARA` in this study is topologically identical to the model
trained in the final LARA study:

* B=32 stem, `T = L` (no bottleneck truncation), six-level **pyramid**
  upsampling decoder,
* a two-layer refinement head instead of a single convolution,
* `latent_dim = 1500 // cr`, so the compression ratio is exact by construction
  and then verified by the pre-flight below.
""")

MD[11] = md("""## CR pre-flight

Builds all 50 arm/CR combinations and measures each latent width directly -
never assuming the requested CR - so a dimension bug fails in seconds instead
of after the full run. Raises if any combination is more than
`CR_TOL = 0.001` (0.1%) off, or if the latent is not exactly `1500 // cr`.
""")

MD[13] = md("""## Training and evaluation

The trainer is the original uniform-budget function with every literal pinned
to the `CONFIG` block: Adam `lr=2e-4`, `weight_decay=1e-5`, gradient clip 1.0,
`ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6)`, early stopping 8,
120 epochs max. Early stopping decides the effective duration, so all five arms
get the same budget.

Evaluation reports **per-sample** statistics, matching `Final_LARA_27092026`:
SNR is `10*log10(power_i / mse_i)` per trace and then averaged, alongside
`SNR_median` and the 25th/75th percentiles. There is no Wavelet branch because
none of the five arms is a wavelet model, and `Reconstructions` is stripped
before the CSV is written.
""")

MD[16] = md("""## Figures and the main run

The figure code is verbatim from the original study. It takes its labels from
the `Model` column of the results frame, so the legends, per-model folders,
titles and filenames all read `Light_Capacity_LARA` with no plotting edit.

`main()` then reports, for every arm: achieved CR against requested CR, a
summary table, parameter efficiency, and a paired sign test of median SNR
against `Heavy_Capacity_HARA` across the ten matched CRs.

```python
# CPU smoke test first (100 traces, 1 CR, 2 epochs, all five arms)
import os; os.environ["FINAL_ABLATION_SMOKE"] = "1"
```

Expect roughly 18 h on an RTX 4080 SUPER.
""")

# ---------------------------------------------------------------- assemble
def code(t):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": t.splitlines(True)}


cells = [
    MD[0],
    code(read("n01_setup.py")),
    MD[2],
    code(read("n03_data.py")),
    code(S[4]),      # EfficientSeismicDataset            - verbatim
    {"cell_type": "markdown", "metadata": {}, "source": A[5].splitlines(True)},
    code(S[6]),      # AttentionBlock / blocks            - verbatim
    MD[7],
    code(S[8]),      # the four HARA arms                 - verbatim
    MD[9],
    code(n10),       # class LARA verbatim + arm registry
    MD[11],
    code(read("n12_preflight.py")),
    MD[13],
    code(tr),        # trainer, literals pinned to CONFIG
    code(n15),        # metric helpers verbatim + new per-sample evaluate_model
    MD[16],
    code(S[14]),     # plotting                           - verbatim
    code(read("n18_main.py")),
]

nb = {"cells": cells, "metadata": {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.11.11"}},
    "nbformat": 4, "nbformat_minor": 5}

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("trainer patch log:")
for r in report:
    print(r)
print(f"\nwrote {OUT}")
print(f"  cells: {len(cells)}  (code {sum(1 for c in cells if c['cell_type']=='code')}, "
      f"markdown {sum(1 for c in cells if c['cell_type']=='markdown')})")
print(f"  class LARA: {len(lara_class)} chars, sha256 "
      f"{__import__('hashlib').sha256(lara_class.encode()).hexdigest()[:16]}")
arm_names = re.findall(r'^\s+\("([A-Za-z_]+)",', n10, re.M)
print(f"  arms: {arm_names}")
