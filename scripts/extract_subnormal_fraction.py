"""Recompute the subnormal-weight fraction of every trained checkpoint.

The benchmark script `Final_LARA_compute_bench_27092026.py` prints this while
timing the models, but it needs a GPU and the dataset. The quantity depends only
on the stored weights, so it can be recomputed from the checkpoints alone. This
script writes `results/subnormal_fraction.csv` so the claim in Section 2.10 of
the paper can be checked without a GPU.

    python scripts/extract_subnormal_fraction.py
"""
import csv
import os
import sys

import torch

TINY = 1.1754943508222875e-38  # smallest normal float32

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CRS = [2, 3, 5, 10, 15, 20, 30, 50, 60, 100]
MODELS = ["LARA", "GeneralizedAutoencoder", "AE_PureConcat"]


def ckpt_dir():
    for rel in (r"seismic 26 9 2026\ckpts", r"ckpts"):
        p = os.path.join(ROOT, rel)
        if os.path.isdir(p):
            return p
    raise FileNotFoundError("no ckpts/ directory found under %s" % ROOT)


def fraction(state):
    n = tot = 0
    for v in state.values():
        if not torch.is_tensor(v) or not v.is_floating_point():
            continue
        f = v.detach().abs().flatten().to(torch.float64)
        n += f.numel()
        tot += int((f < TINY).sum())
    return tot / n, n


def main():
    cd = ckpt_dir()
    rows = []
    for name in MODELS:
        for cr in CRS:
            p = os.path.join(cd, f"best_{name}_cr{cr}.pth")
            if not os.path.exists(p):
                print("  skip %-24s CR=%-4d (no checkpoint)" % (name, cr))
                continue
            state = torch.load(p, map_location="cpu", weights_only=True)
            frac, n = fraction(state)
            rows.append(dict(Model=name, CR=cr, Parameters=n,
                             Subnormal_Count=int(round(frac * n)),
                             Subnormal_Fraction=round(frac, 6),
                             Subnormal_Percent=round(100 * frac, 2)))
            print("  %-24s CR=%-4d %10d params  %6.2f%% subnormal"
                  % (name, cr, n, 100 * frac))
    out = os.path.join(ROOT, "results", "subnormal_fraction.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("\nwrote %s (%d rows)" % (out, len(rows)))
    lara = [r for r in rows if r["Model"] == "LARA"]
    hi = [r for r in lara if r["CR"] >= 50]
    if hi:
        print("LARA at CR>=50: %.1f%% to %.1f%% subnormal"
              % (min(r["Subnormal_Percent"] for r in hi),
                 max(r["Subnormal_Percent"] for r in hi)))
    other = [r for r in rows if r["Model"] != "LARA"]
    if other:
        print("baselines        : %.2f%% to %.2f%% subnormal"
              % (min(r["Subnormal_Percent"] for r in other),
                 max(r["Subnormal_Percent"] for r in other)))
    return 0


if __name__ == "__main__":
    sys.exit(main())