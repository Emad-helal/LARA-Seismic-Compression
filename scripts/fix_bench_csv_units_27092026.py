"""One-time unit fix for the two published compute_bench CSVs.

WHAT WAS WRONG
--------------
`timeit` measures ONE forward call, and that call carries a whole batch of
`batch` traces. The value was stored in a column named `ms_per_trace` and the
three derived columns were then computed from it without dividing by the batch
size. On the batch-32 rows that made `traces_per_second` a batches-per-second
figure, which overstated throughput by 32x, and carried the same factor of 32
into `stations_supported` and `realtime_factor`. The stored maxima were
14,999,886 for `realtime_factor` on the CPU file.

The coder stages are timed one trace at a time and carry no `batch` value, so
they were already correct. The batch-1 rows were already correct. Nothing else
in either file changes: no measured value is altered, and the raw call time is
retained in a new `ms_per_batch_call` column.

The same arithmetic is now done at source in
Final_LARA_compute_bench_27092026.py, so a fresh benchmark run produces these
columns correctly without this script.

READING realtime_factor / stations_supported
--------------------------------------------
These are meaningful only for a whole pipeline, not for one `stage` row. Two
pre-existing measurement limits make single-stage maxima meaningless: the
`dequantize` stage measures exactly 1 us, which is the `perf_counter` floor on
this host, and the `AE_PureConcat` decoder at CR >= 50 is a single
Linear(latent -> 1500), so a batch-32 call really is ~25 us, or 0.8 us per
trace. Either row therefore yields a six-figure `stations_supported` that says
nothing about the codec. The document builders compute capacity from the summed
pipeline cost instead; this script does not attempt to fix the floor, which would
require re-measuring.

Usage:  python fix_bench_csv_units_27092026.py [--dry-run]
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

RUN = (r"D:\Post_Doctor\Dr. Mostafa\Enhanced_Residual_AutoEncoder\Dr. Omar"
       r"\Review_27092026\seismic 26 9 2026")
COST = os.path.join(RUN, "results", "compute_cost")
CHANNELS_PER_STATION = 3
SIGNAL_SECONDS = 1500 / 100.0

DERIVED = ["traces_per_second", "stations_supported", "realtime_factor"]
TIMED = ["ms_per_trace", "ms_p25", "ms_p75"]
MEASURED = ["model", "cr", "latent_dim", "batch", "threads", "stage",
            "denormal_mode", "nparam", "ckpt_bytes",
            "subnormal_fraction", "precision", "peak_infer_bytes",
            "range_source", "coded_bits", "quantiser_bits", "encoder_flops",
            "decoder_flops"]
HW = [c for c in ("hw_platform", "hw_python", "hw_torch", "hw_logical_cores",
                  "hw_physical_cores", "hw_ram_gb", "hw_cpu", "hw_accelerator",
                  "hw_accelerator_kind", "hw_capability", "hw_cuda_version",
                  "hw_device_memory_gb", "hw_hardware_id")
      if c in pd.read_csv(glob.glob(os.path.join(COST, "**", "compute_bench_*_full.csv"),
                                    recursive=True)[0], nrows=0).columns]


def fix(df):
    """Return (new_df, report). Only the timed columns and the three derived
    columns move; the raw call times are retained under *_per_batch_call."""
    bs = df["batch"].fillna(1.0)
    raw = {c: df[c].copy() for c in TIMED}
    tim = raw["ms_per_trace"].notna() & (raw["ms_per_trace"] > 0)

    out = df.copy()
    for c in TIMED:
        out[c] = raw[c] / bs
    out["traces_per_second"] = np.where(tim, 1000.0 / out["ms_per_trace"], np.nan)
    out["stations_supported"] = out["traces_per_second"] / (CHANNELS_PER_STATION / SIGNAL_SECONDS)
    out["realtime_factor"] = (SIGNAL_SECONDS * 1000.0) / out["ms_per_trace"]
    for c in TIMED:
        out[f"{c}_per_batch_call"] = raw[c]

    # Column order: every original column in place, raw call times appended.
    out = out[[c for c in df.columns] + [f"{c}_per_batch_call" for c in TIMED]]

    rep = {"rows": len(df), "b1_unchanged": 0, "b32_unchanged": 0,
           "b1_rows": 0, "b32_rows": 0}
    is_b1 = (bs == 1.0)
    is_b32 = (df["batch"] == 32.0)
    rep["b1_rows"] = int(is_b1.sum())
    rep["b32_rows"] = int(is_b32.sum())
    rep["b1_unchanged"] = int(all(np.array_equal(
        out.loc[is_b1, c].values, df.loc[is_b1, c].values, equal_nan=True)
        for c in TIMED))
    rep["b32_unchanged"] = int(all(np.allclose(
        out.loc[is_b32, c].values, df.loc[is_b32, c].values / 32.0,
        rtol=1e-12, equal_nan=True) for c in TIMED))
    return out, rep


def main():
    dry = "--dry-run" in sys.argv
    files = sorted(glob.glob(os.path.join(COST, "compute_cost_*",
                                          "compute_bench_*_full.csv")))
    if not files:
        raise SystemExit(f"no compute_bench_*_full.csv under {COST}")
    for path in files:
        before = pd.read_csv(path)
        after, rep = fix(before)

        # Hard assertions: nothing measured may move.
        for c in MEASURED + HW:
            if c in before.columns:
                b, a = before[c], after[c]
                if c in ("cr", "batch", "threads", "latent_dim"):
                    ok = b.equals(a)
                else:
                    ok = np.allclose(b.astype("float64", errors="ignore"),
                                     a.astype("float64", errors="ignore"),
                                     rtol=1e-12, atol=0, equal_nan=True) \
                        if b.dtype.kind in "fiu" else b.equals(a)
                if not ok:
                    raise SystemExit(f"ABORT {os.path.basename(path)}: column {c} changed")
        for c in ("nparam", "ckpt_bytes", "subnormal_fraction"):
            if c in before.columns and not np.allclose(
                    before[c].astype("float64"), after[c].astype("float64"),
                    rtol=1e-12, atol=0, equal_nan=True):
                raise SystemExit(f"ABORT {os.path.basename(path)}: {c} must not move")
        if rep["b1_unchanged"] != 1:
            raise SystemExit(f"ABORT {os.path.basename(path)}: batch-1 rows changed")

        tag = "DRY-RUN" if dry else "writing"
        print(f"{tag} {os.path.basename(path)}")
        print(f"  rows {rep['rows']} (batch-1 {rep['b1_rows']}, batch-32 {rep['b32_rows']})")
        print(f"  batch-1 ms_per_trace unchanged: {bool(rep['b1_unchanged'])}")
        print(f"  batch-32 ms_per_trace == old/32: {bool(rep['b32_unchanged'])}")
        for c in DERIVED:
            ob = before[c].max()
            nb = after[c].max()
            print(f"  {c:22s} max {ob:>14,.2f} -> {nb:>14,.2f}")
        if not dry:
            after.to_csv(path, index=False)
            print(f"  saved {os.path.getsize(path):,} bytes")


if __name__ == "__main__":
    main()
