"""Computational cost benchmark for the neural codecs of Final_LARA_27092026.

Measures, per model and compression ratio: FLOPs and MACs of encoder and decoder
separately, latency and throughput under several batch sizes and thread counts,
the cost of the full codec path including quantization and entropy coding,
memory, and the real-time factor.

PORTABILITY
-----------
The script detects the hardware it runs on and stamps a fingerprint on every
output row, so a CPU run and a GPU run can never be silently confused. Results
are written to a per-hardware folder, compute_cost_CPU/ or compute_cost_GPU/, so
each folder holds exactly one platform and neither can pick up the other's files.

UNITS
-----
`ms_per_trace` is latency per TRACE, not per forward call. One timed call carries
a whole batch, so the per-trace value is the call time divided by the batch size;
the raw call time is kept in `ms_per_batch_call`. The coder stages are timed one
trace at a time. The three derived columns are computed from the per-trace value.

THREADS
-------
Each thread count is measured in a SEPARATE PROCESS. Switching the OpenMP pool
inside one process, while a large array is resident, inflates single-thread
latency by an order of magnitude on this hybrid Intel part; only a clean process
per configuration gives numbers that can be trusted. The parent therefore
spawns one child per thread count.

FLOP CONVENTION
---------------
A multiply-add counts as 2 FLOPs. Matrix multiplications and convolutions are
counted; elementwise operations (activations, residual additions, normalisation)
are not. This is the usual convention and is stated as a limitation.

Usage
-----
    python Final_LARA_compute_bench_27092026.py
    python Final_LARA_compute_bench_27092026.py --quick
    python Final_LARA_compute_bench_27092026.py --no-codec
"""

import os
import sys
import re
import tempfile
import gc
import json
import time
import math
import argparse
import hashlib
import platform
import warnings

warnings.filterwarnings("ignore")
import logging
logging.getLogger("matplotlib").setLevel(logging.ERROR)

import numpy as np
import pandas as pd
import torch
import psutil

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

RUN_DIR = (r"D:\Post_Doctor\Dr. Mostafa\Enhanced_Residual_AutoEncoder\Dr. Omar"
           r"\Review_27092026\seismic 26 9 2026")
CKPT_DIR = os.path.join(RUN_DIR, "ckpts")
COST_ROOT = os.path.join(RUN_DIR, "results", "compute_cost")
NOTEBOOK = r"C:\Users\master\Final_HARA\Seismic 26092025\Final_LARA_27092026.ipynb"

MODELS = ["LARA", "GeneralizedAutoencoder", "AE_PureConcat"]
CRS = [2, 3, 5, 10, 15, 20, 30, 50, 60, 100]
EXPECTED_LARA_SHA = "ea4c8f4b8a2caaa3"
N_SAMPLES = 1500
SAMPLE_RATE_HZ = 100.0
CHANNELS_PER_STATION = 3
SIGNAL_SECONDS = N_SAMPLES / SAMPLE_RATE_HZ
CODER_TRACES = 50
CODER_BITS = 5
MIN_SECONDS = 0.5
CODER_SECONDS = 0.2


# ===================================================== hardware fingerprint
def out_dir(hw):
    """Deliverables go to a per-hardware folder: compute_cost_CPU/ holds only
    CPU results, compute_cost_GPU/ only GPU results. The worker subprocess
    recomputes its own fingerprint, so it resolves the same folder."""
    return os.path.join(COST_ROOT, f"compute_cost_{hw['accelerator_kind']}")


def hardware_fingerprint():
    info = {"platform": platform.platform(), "python": platform.python_version(),
            "torch": torch.__version__}
    try:
        info["logical_cores"] = psutil.cpu_count(logical=True)
        info["physical_cores"] = psutil.cpu_count(logical=False)
        info["ram_gb"] = round(psutil.virtual_memory().total / 2 ** 30, 1)
    except Exception:
        pass
    cpu = platform.processor() or ""
    if sys.platform == "win32":
        try:
            import subprocess
            o = subprocess.run(["powershell", "-NoProfile", "-Command",
                                "(Get-CimInstance Win32_Processor).Name"],
                               capture_output=True, text=True, timeout=30)
            if o.returncode == 0 and o.stdout.strip():
                cpu = o.stdout.strip().splitlines()[0].strip()
        except Exception:
            pass
    info["cpu"] = cpu
    if torch.cuda.is_available():
        i = torch.cuda.current_device()
        cap = torch.cuda.get_device_capability(i)
        info["accelerator"] = torch.cuda.get_device_name(i)
        info["accelerator_kind"] = "gpu"
        info["capability"] = f"{cap[0]}.{cap[1]}"
        info["cuda_version"] = torch.version.cuda
        info["device_memory_gb"] = round(
            torch.cuda.get_device_properties(i).total_memory / 2 ** 30, 1)
    else:
        info["accelerator"] = "none"
        info["accelerator_kind"] = "cpu"
    slug_src = info["accelerator"] if info["accelerator_kind"] == "gpu" else cpu
    slug = re.sub(r"[^A-Za-z0-9]+", "_", slug_src).strip("_").lower()
    info["hardware_id"] = f"{slug[:44]}_{info['accelerator_kind']}"
    return info


# ================================================================ timing util
def timeit(fn, device, warmup, min_seconds=0.5, calib=5):
    """Median ms per call over a run of at least `min_seconds` wall clock.

    A fixed ITERATION COUNT is unsafe here. This host is a throttled shared VM
    (WMI reports 1400 MHz for a part whose base clock is ~2.2 GHz), so the OS
    enforces CPU quota in periodic stalls. A 30-iteration burst of a 4 ms kernel
    lasts ~120 ms and can fall entirely inside a stall, inflating the result by
    20x or more, which is exactly the erratic pattern first observed. Timing a
    fixed DURATION averages over many quota cycles instead.
    """
    cuda = device.type == "cuda"
    for _ in range(warmup):
        fn()
    if cuda:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(calib):
        fn()
    if cuda:
        torch.cuda.synchronize()
    per = max((time.perf_counter() - t0) / calib, 1e-9)
    n = max(20, min(20000, int(min_seconds / per)))
    ts = np.empty(n)
    t_start = time.perf_counter()
    for i in range(n):
        if cuda:
            torch.cuda.synchronize()
        a = time.perf_counter()
        fn()
        if cuda:
            torch.cuda.synchronize()
        ts[i] = (time.perf_counter() - a) * 1000.0
    wall = time.perf_counter() - t_start
    return (float(np.median(ts)), float(np.percentile(ts, 25)),
            float(np.percentile(ts, 75)), int(n), wall)


# ============================================================== source models
def notebook_code():
    with open(NOTEBOOK, encoding="utf-8") as fh:
        nb = json.load(fh)
    return {i: "".join(c["source"]) for i, c in enumerate(nb["cells"])
            if c["cell_type"] == "code"}


def build_namespace(with_data=False):
    code = notebook_code()
    lara = code[6]
    i0 = lara.index("class LARA(nn.Module)")
    i1 = lara.find("\nclass ", i0 + 10)
    sha = hashlib.sha256(((lara[i0:i1] if i1 > 0 else lara[i0:]).rstrip())
                         .encode()).hexdigest()[:16]
    if sha != EXPECTED_LARA_SHA:
        raise SystemExit(f"ABORT: class LARA hashes to {sha}, expected {EXPECTED_LARA_SHA}")
    ns = {"__name__": "bench"}
    exec(compile(code[1], "<cell 1>", "exec"), ns)
    ns.update({"N_TRACES": None, "COMPONENT": 2, "N_EVAL": 200, "MIN_MAGNITUDE": 2.5,
               "MAX_DISTANCE_KM": 60, "CSV_PATH": None, "H5_PATH": None, "BATCH": 16,
               "WORKERS": 0, "SEED": 42, "PIN_MEMORY": False,
               "device": torch.device("cpu")})
    ns["MAIN_MODEL_NAME"] = "LARA"
    ns["LARA_VARIANT"] = "pyramid_funnel32"
    for ci in (3, 5, 6):
        exec(compile(code[ci], f"<cell {ci}>", "exec"), ns)
    ns["_lara_sha"] = sha
    if with_data:
        ns["CSV_PATH"] = r"D:\STEAD\merged.csv"
        ns["H5_PATH"] = r"D:\STEAD\merged.hdf5"
        exec(compile(code[4], "<cell 4>", "exec"), ns)
    return ns


def build_model(ns, name, cr, device):
    m = (ns["LARA"](cr, variant="pyramid_funnel32") if name == "LARA" else ns[name](cr))
    state = torch.load(os.path.join(CKPT_DIR, f"best_{name}_cr{cr}.pth"),
                       map_location="cpu", weights_only=True)
    m.load_state_dict(state)
    return m.to(device).eval(), subnormal_fraction(state)


def subnormal_fraction(state):
    """Fraction of float32 parameters that are subnormal (|v| < 1.1755e-38).

    Subnormal operands take a microcoded path on x86 and can cost up to two
    orders of magnitude, so this fraction predicts the CPU inference cost far
    better than parameter count does.
    """
    lo, n, sub = 1.1754944e-38, 0, 0
    for _, v in state.items():
        if not torch.is_floating_point(v):
            continue
        a = v.detach().double().abs().flatten()
        n += a.numel()
        sub += int(((a > 0) & (a < lo)).sum())
    return sub / n if n else 0.0


def enc_fn(model, name):
    if name in ("LARA", "GeneralizedAutoencoder"):
        return lambda t: model.encoder(t)
    return lambda t: torch.cat([b(t) for b in model.branches], dim=1)


def dec_fn(model, name):
    if name == "LARA":
        return lambda z: model.head(model.coarse_decoder(z))
    return lambda z: model.decoder(z)


# ================================================================== worker
def run_worker(args, nthreads):
    torch.set_num_threads(nthreads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = MODELS[:1] if args.quick else MODELS
    crs = CRS[:3] if args.quick else CRS
    batches = [1, 32] if not args.quick else [1]
    warm = 5 if args.quick else 10

    ns = build_namespace()
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from arithcoder import AdaptiveCoder
    if args.stage == "flops":
        from torch.utils.flop_counter import FlopCounterMode

    calib = None
    if not args.no_codec and args.calib and os.path.exists(args.calib):
        calib = torch.tensor(np.load(args.calib))

    rows = []

    def add(**kw):
        kw.setdefault("precision", "float32")
        kw.setdefault("ms_per_trace", np.nan)
        kw.setdefault("ms_p25", np.nan)
        kw.setdefault("ms_p75", np.nan)
        rows.append(kw)

    for name in models:
        for cr in crs:
            lat = N_SAMPLES // cr
            ck = os.path.join(CKPT_DIR, f"best_{name}_cr{cr}.pth")
            cks = os.path.getsize(ck) if os.path.exists(ck) else -1
            model, subfrac = build_model(ns, name, cr, device)
            npar = sum(p.numel() for p in model.parameters())
            e, d = enc_fn(model, name), dec_fn(model, name)
            print(f"[{name} CR={cr}] subnormal weights = {subfrac:.2%}")

            # Network stages are timed BOTH as deployed (denormals active) and
            # with subnormals flushed, because the difference is the single
            # largest term in this measurement and is a deployment decision.
            for mode in (("as_deployed", "flushed") if args.stage == "timing" else ()):
                torch.set_flush_denormal(mode == "flushed")
                for bs in batches:
                    x = torch.randn(bs, 1, N_SAMPLES, device=device, dtype=torch.float32)
                    zc = torch.randn(bs, lat, device=device, dtype=torch.float32)
                    with torch.inference_mode():
                        m1, a1, b1, n1, _ = timeit(lambda: e(x), device, warm, MIN_SECONDS)
                        add(model=name, cr=cr, latent_dim=lat, batch=bs, threads=nthreads,
                            stage="encoder", denormal_mode=mode, ms_per_trace=m1,
                            ms_p25=a1, ms_p75=b1, nparam=npar, ckpt_bytes=cks,
                            subnormal_fraction=subfrac)
                        m2, a2, b2, n2, _ = timeit(lambda: d(zc), device, warm, MIN_SECONDS)
                        add(model=name, cr=cr, latent_dim=lat, batch=bs, threads=nthreads,
                            stage="decoder", denormal_mode=mode, ms_per_trace=m2,
                            ms_p25=a2, ms_p75=b2, nparam=npar, ckpt_bytes=cks,
                            subnormal_fraction=subfrac)
            torch.set_flush_denormal(True)

            if args.stage == "timing":
                # memory
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                    with torch.inference_mode():
                        model(torch.randn(32, 1, N_SAMPLES, device=device))
                    peak = int(torch.cuda.max_memory_allocated())
                else:
                    proc = psutil.Process()
                    base = proc.memory_info().rss
                    xb = torch.randn(32, 1, N_SAMPLES)
                    with torch.inference_mode():
                        for _ in range(4):
                            model(xb)
                    peak = max(int(proc.memory_info().rss - base), 0)
                add(model=name, cr=cr, latent_dim=lat, batch=32,
                    threads=nthreads, stage="memory", nparam=npar, ckpt_bytes=cks,
                    peak_infer_bytes=peak, subnormal_fraction=subfrac)
                del model
                gc.collect()

            # FLOPs, only in the dedicated flops process
            if args.stage == "flops":
                m32, _sf = build_model(ns, name, cr, torch.device("cpu"))
                e32, d32 = enc_fn(m32, name), dec_fn(m32, name)
                x1 = torch.randn(1, 1, N_SAMPLES)
                with torch.inference_mode():
                    z1 = e32(x1)
                    with FlopCounterMode(display=False) as fc:
                        e32(x1)
                    fe = fc.get_total_flops()
                    with FlopCounterMode(display=False) as fc:
                        d32(z1)
                    fd = fc.get_total_flops()
                add(model=name, cr=cr, latent_dim=lat, batch=1, threads=nthreads,
                    stage="flops", nparam=sum(p.numel() for p in m32.parameters()),
                    ckpt_bytes=cks, encoder_flops=float(fe), decoder_flops=float(fd))
                del m32
                gc.collect()
                continue

            m32, _sf = build_model(ns, name, cr, torch.device("cpu"))
            e32 = enc_fn(m32, name)

            # full codec path
            if (not args.no_codec) and calib is not None:
                with torch.inference_mode():
                    zc = e32(calib)
                zn = zc.numpy().astype(np.float64)
                lo = zn.min(axis=0)
                span = np.maximum(zn.max(axis=0) - lo, 1e-12)
                A = 1 << CODER_BITS
                delta = span / (A - 1)
                q = np.clip(np.rint((zn - lo) / delta), 0, A - 1).astype(np.int64)
                # encode() returns (num_bits, decoder); the decoder is a live
                # stream object, so a round trip is timed with a fresh pair
                coder = AdaptiveCoder(A)
                pairs = [coder.encode(q[t].tolist()) for t in range(q.shape[0])]
                mean_bits = float(np.mean([nb for nb, _ in pairs]))
                npar32 = sum(p.numel() for p in m32.parameters())

                def roundtrip():
                    nd, dd = AdaptiveCoder(A).encode(q[0].tolist())
                    return AdaptiveCoder(A).decode(dd, lat)

                for stage, fn in (
                    ("quantize", lambda: np.clip(np.rint((zn - lo) / delta), 0, A - 1)),
                    ("entropy_encode", lambda: AdaptiveCoder(A).encode(q[0].tolist())),
                    ("entropy_roundtrip", roundtrip),
                    ("dequantize", lambda: lo + q[0] * delta),
                ):
                    m_, a_, b_, n_, w_ = timeit(fn, device, max(2, warm // 2), CODER_SECONDS)
                    add(model=name, cr=cr, latent_dim=lat, threads=nthreads, stage=stage,
                        ms_per_trace=m_, ms_p25=a_, ms_p75=b_,
                        nparam=npar32, ckpt_bytes=cks,
                        range_source="calibrated", coded_bits=mean_bits,
                        quantiser_bits=CODER_BITS)
                del pairs
                gc.collect()
    return pd.DataFrame(rows)


# ==================================================================== main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-codec", action="store_true")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--calib", default="")
    ap.add_argument("--stage", default="timing", choices=["timing", "flops"])
    ap.add_argument("--codec-traces", type=int, default=CODER_TRACES)
    args = ap.parse_args()

    hw = hardware_fingerprint()
    OUT_DIR = out_dir(hw)
    os.makedirs(OUT_DIR, exist_ok=True)

    if args.worker:
        if args.threads:
            torch.set_num_threads(args.threads)
        df = run_worker(args, args.threads or torch.get_num_threads())
        out = os.path.join(OUT_DIR, f"_worker_{args.stage}_t{args.threads}.csv")
        df.to_csv(out, index=False)
        print(f"worker {args.stage} t={args.threads} -> {out} ({len(df)} rows)")
        return

    print("=" * 78)
    print("Computational cost benchmark - Final_LARA_27092026")
    print("=" * 78)
    for k, v in hw.items():
        print(f"  {k:<20}: {v}")
    print(f"[guard] class LARA sha256 {build_namespace()['_lara_sha']} -> matches the trained architecture")

    # calibration traces are loaded ONCE here and handed to the workers as a
    # small array, so no worker pays the 25,000-trace load and no worker runs
    # with that memory resident while timing
    calib_path = ""
    if not args.no_codec:
        try:
            print("\n[calib] loading evaluation traces once ...")
            dns = build_namespace(with_data=True)
            arr = np.asarray(dns["test_data"][:args.codec_traces], dtype=np.float32)
            del dns
            gc.collect()
            # Scratch, deliberately NOT written into OUT_DIR: the deliverable
            # folder must hold results only, and a 300 KB .npy there reads as
            # an output it is not.
            calib_path = os.path.join(
                tempfile.gettempdir(), f"_calib_{hw['hardware_id']}.npy")
            np.save(calib_path, arr)
            print(f"[calib] {arr.shape[0]} traces -> scratch {calib_path} "
                  f"({os.path.getsize(calib_path):,} bytes)")
        except Exception as e:
            print(f"[calib] unavailable ({type(e).__name__}: {e}); "
                  f"coder stages will be omitted")

    settings = sorted({1, torch.get_num_threads()})
    if args.quick:
        settings = settings[:1]
    print(f"\nspawning {len(settings)} worker process(es), threads={settings}")
    import subprocess
    frames = []
    jobs = [("timing", nt) for nt in settings] + [("flops", 1)]
    for stage, nt in jobs:
        cmd = [sys.executable, os.path.abspath(__file__), "--worker",
               "--stage", stage, "--threads", str(nt), "--calib", calib_path]
        if args.quick:
            cmd.append("--quick")
        if args.no_codec:
            cmd.append("--no-codec")
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stdout[-3000:]); print(r.stderr[-2000:])
            raise SystemExit(f"worker {stage} threads={nt} failed")
        frames.append(pd.read_csv(os.path.join(OUT_DIR, f"_worker_{stage}_t{nt}.csv")))
        print(f"  {stage} t={nt}: {len(frames[-1])} rows")

    df = pd.concat(frames, ignore_index=True)
    for k, v in hw.items():
        df[f"hw_{k}"] = v
    tim = df["ms_per_trace"].notna() & (df["ms_per_trace"] > 0)
    # `timeit` measures ONE forward call, and that call carries a whole batch of
    # `batch` traces, so the per-trace latency is the call time divided by the
    # batch size. The coder stages are timed one trace at a time and carry no
    # batch value, so they divide by 1. Without this the three derived columns
    # below report batches per second on the batch-32 rows and overstate
    # throughput by the batch size.
    bs = df["batch"].fillna(1.0)
    df["ms_per_batch_call"] = df["ms_per_trace"]
    per_trace_ms = df["ms_per_trace"] / bs
    df["ms_per_trace"] = per_trace_ms
    df["traces_per_second"] = np.where(tim, 1000.0 / per_trace_ms, np.nan)
    df["stations_supported"] = df["traces_per_second"] / (CHANNELS_PER_STATION / SIGNAL_SECONDS)
    df["realtime_factor"] = (SIGNAL_SECONDS * 1000.0) / per_trace_ms

    tag = "quick" if args.quick else "full"
    csv = os.path.join(OUT_DIR, f"compute_bench_{hw['hardware_id']}_{tag}.csv")
    df.to_csv(csv, index=False)
    hjson = os.path.join(OUT_DIR, f"hardware_{hw['hardware_id']}.json")
    with open(hjson, "w", encoding="utf-8") as fh:
        json.dump(hw, fh, indent=2)
    for stage, nt in jobs:
        p = os.path.join(OUT_DIR, f"_worker_{stage}_t{nt}.csv")
        if os.path.exists(p):
            os.remove(p)
    print(f"\nwrote {csv}  ({len(df)} rows, {os.path.getsize(csv):,} bytes)")
    print(f"wrote {hjson}")
    print(f"\nhardware_id = {hw['hardware_id']}")
    print(f"folder      = {OUT_DIR}")
    print("Run the identical command on the GPU machine to populate "
          "compute_cost_GPU/.")


if __name__ == "__main__":
    main()
