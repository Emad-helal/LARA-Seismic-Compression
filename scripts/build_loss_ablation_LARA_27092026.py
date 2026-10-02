"""Loss-function ablation on the FINAL LARA model -- answers Reviewer 2, Comment 4.

    "The HARA model is trained using the mean squared error (MSE) reconstruction
     loss. However, the manuscript claims that the method preserves important
     seismic features, including P- and S-wave arrivals, phase information, and
     frequency-domain characteristics. Have the authors investigated
     frequency-domain, phase-aware, or other seismic-specific loss functions? An
     additional comparison or ablation study would strengthen the justification
     for using MSE alone."

WHAT THIS IS
    Seven training objectives x ten compression ratios = 70 runs, every one of
    them from scratch. Only the loss changes; the architecture, the data, the
    split, the hyper-parameters and the initial weights are identical across the
    seven arms at a given ratio. Each arm therefore isolates the effect of one
    loss function and nothing else.

WHY IT SUPERSEDES `seismic 21 9 2026`
    That study ran the same seven-arm design on
    `LightCapacityResidualAutoencoder` -- the light-capacity architecture that
    PREDECEDED LARA. It shares the 32-channel stem, the channel attention block
    and the residual blocks, but differs in the bottleneck (T = min(z, L) rather
    than T = L, and a fixed 16-channel width), the decoder (linear upsampling at
    a constant 16-channel width rather than a six-level channel pyramid ramping
    to 64), and the output head (a single convolution rather than a two-layer
    refinement). The parameter counts differ by 1.02x at CR=2 and 3.80x at CR=20.

    So that study bounds the effect of the loss on the PREVIOUS network. It does
    not answer the reviewer's question about the model the paper actually
    proposes. This script re-runs it on LARA. `seismic 21 9 2026` is left
    untouched as the provenance record.

    Note that the ratio is not uniform across CR: the funnel engages only for
    cr >= 10, so at CR = 2, 3 and 5 the two models are close (1.02x-1.05x) while
    at CR = 10 and 20 they are 2.43x and 3.80x apart.

ARCHITECTURE FIDELITY
    `class LARA` is embedded VERBATIM below (10,736 characters, sha256
    ea4c8f4b8a2caaa3eadc31e7db5bdc37fe5ca958b8253df629918480b463970a, taken
    from cell 6 of `seismic 26 9 2026/Final_LARA_27092026.ipynb`). The same
    truncated sha256 is asserted by `Final_LARA_bps_analysis_27092026.py` and
    `Final_LARA_compute_bench_27092026.py`. `verify_lara_hash()` re-checks it at
    import, so an accidental edit to the class aborts the run instead of quietly
    producing results for a different topology.

    Pass `--from-notebook <path>` to re-derive the class from a notebook at run
    time and cross-check it against the embedded copy instead.

TWO FIXES RELATIVE TO `seismic 21 9 2026`
    1. `TorchWaveletDecompose` hard-coded `pywt.Wavelet('db4')` inside
       `_db4_filters` and therefore ignored its own `wavelet` argument. The name
       is now passed through. This study only ever uses 'db4', so the numerics
       are bit-identical and no result changes.
    2. `lambda_ncc` was an unrecorded `make_loss` default of 0.3, which meant the
       winning arm could not be reproduced from the saved CSVs alone. It is now
       `LAMBDA_NCC` in CONFIG and is written to the selection CSV.

TWO DIAGNOSTICS DELIBERATELY NOT REPORTED
    * `PolarityMatch`. Under this preprocessing it cannot return anything but
      1.0. `EfficientSeismicDataset` min-max scales every trace to [0, 1], so no
      input sample is ever negative; the decoder ends in `Sigmoid`, so no output
      sample is ever negative either. `_first_motion_polarity` returns
      `1 if mean(seg) >= 0 else -1`, so the polarity of both traces is always
      +1 and the metric is a constant. It is arithmetically correct and
      meaningless. In the predecessor study it also silently disabled the
      documented "tie-break on polarity" rule.
    * `PickErr_samples`. It is the raw global `argmax` of the STA/LTA ratio over
      the whole 1,500-sample window -- no threshold crossing and no pre-trigger
      window. It returned between 37 and 384 samples for the SAME MSE control as
      CR increased, i.e. up to 26 % of the record. That is a statistic about the
      largest local energy fluctuation of a smooth reconstructed baseline, not an
      arrival-time error.

TWO DEFECTS PRESERVED VERBATIM, AND REPORTED AS FINDINGS
    * `stft_phase_loss` is identically blind to a global polarity inversion. If
      x_hat = -x then Y_hat = -X, so arg Y_hat = arg X + pi uniformly, and that
      constant offset cancels exactly in the frame-to-frame phasor difference.
      The loss is therefore 0 for a perfectly polarity-inverted reconstruction.
      The arms named `STFT_Phase` and `STFT_Phase_Arrival` do NOT test polarity
      preservation despite their names.
    * `Arrival_Floor_eff` scales the arrival-weight floor from 1.0 at CR=2 to
      3.0 at CR=100. The weight is `floor + ratio/max(ratio)`, so a HIGHER floor
      FLATTENS the contrast between quiet and energetic samples; it does not
      sharpen it. The contrast falls from 2:1 to 4:3. Kept as designed, because
      changing it would change the experiment; disclosed in the manuscript.

THE MSE ARM IS ALSO A REPRODUCTION CHECK
    The control arm trains from scratch, so its six-to-ten runs double as an
    independent reproduction of the published main run. `loss_ablation_LARA_
    mse_reproduction.csv` compares it against the LARA rows of
    `model_comparison_results.csv` at a 0.05 dB tolerance -- the same gate
    `Final_LARA_bps_analysis_27092026.py` already passes on this model and
    split. Warn-only by default so a 0.06 dB drift never wastes a completed run;
    pass `--strict` to abort instead.

RUN
    python build_loss_ablation_LARA_27092026.py --smoke     # ~2 min, all 7 arms
    python build_loss_ablation_LARA_27092026.py              # the real 70 runs

    Always `--smoke` first on the GPU box. It exercises data loading, the model,
    every loss, the trainer, the evaluator and every writer, so a typo cannot
    cost eight hours.

    Runtime for the full grid: roughly 8-9 h on an RTX 4080 SUPER (70 runs at
    ~450 s each, batch 64, early stopping deciding the effective duration), and
    about 500 MB of checkpoints.

OUTPUT  (default `<repo>/seismic_LARA_loss_ablation_results/`)
    loss_ablation_LARA_selection.csv       7 arms x 10 CRs, full metrics + lambdas
    loss_ablation_LARA_percr_deltas.csv    long-form deltas vs the MSE control
    loss_ablation_LARA_percr_dSNR.csv      wide pivot of dSNR
    loss_ablation_LARA_ranking.csv         both metrics + win counts
    loss_ablation_LARA_mse_reproduction.csv  from-scratch MSE vs published LARA
    loss_ablation_LARA_summary.png         2-panel figure, no 3x weighting
    ckpts/best_LARA_cr{CR}_{ARM}.pth       70 checkpoints

WHY THE RANKING REPORTS TWO METRICS
    The predecessor study selected a winner with a CR-weighted mean dSNR that
    put 3x weight on CR = 50 and CR = 100. In that run the winner LOST at both
    of those ratios, which is why its headline value came out negative. A
    weighting that promotes an arm is a poor arbiter when it is tuned against
    the data. This script therefore reports the weighted metric AND the
    unweighted mean, the median, and the raw count of ratios won against the MSE
    control, and the console verdict is written in terms of the unweighted
    numbers. It also warns when `CRs_with_SNR_regression_highCR` is identical
    across all arms, because such a column carries no information.
"""
import os
import sys
import copy
import time
import math
import json
import argparse
import hashlib

import warnings

warnings.filterwarnings("ignore")
import logging

logging.getLogger("matplotlib").setLevel(logging.ERROR)
try:  # the EPS backend chats on every one of the 70+ figure writes
    import warnings as _w

    _w.filterwarnings("ignore", message=".*PostScript backend.*")
except Exception:
    pass

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from skimage.metrics import structural_similarity as ssim
import h5py
import pywt
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass


# ===================================================================== CONFIG
# Everything you need to change for a different machine is in this block.
# The command line can override all of it.
#
# The paths below are the ones the sibling analysis scripts use on this
# workstation (Final_LARA_bps_analysis_27092026.py, Final_LARA_compute_bench_
# 27092026.py). EDIT THEM if the GPU box stores the STEAD files elsewhere.
# =====================================================================
REPO = r"D:\Post_Doctor\Dr. Mostafa\Enhanced_Residual_Autoencoder\Dr. Omar\Review_27092026"

CSV_PATH = r"D:\STEAD\merged.csv"          # STEAD metadata
H5_PATH = r"D:\STEAD\merged.hdf5"          # STEAD waveforms
OUT_DIR = os.path.join(REPO, "seismic_LARA_loss_ablation_results")
SOURCE_NOTEBOOK = r"C:\Users\master\Final_HARA\Seismic 26092025\Final_LARA_27092026.ipynb"
PUBLISHED_CSV = os.path.join(
    REPO, "seismic 26 9 2026", "results", "model_comparison_results.csv"
)

# ---- study geometry (aligned with Final_LARA_27092026) ----
N_TRACES = None          # None = every trace passing the filter (~25,091)
MIN_MAGNITUDE = 2.5      # was 3.0 in the predecessor loss study
MAX_DISTANCE_KM = 60
COMPONENT = 2            # vertical (Z)
N_EVAL = 200             # first N_EVAL traces of the test split, as in the main run
CRS = [2, 3, 5, 10, 15, 20, 30, 50, 60, 100]
HIGH_CR = {50, 100}      # CRs carrying 3x weight in the LEGACY metric only
INPUT_LEN = 1500

# ---- training recipe (identical to Final_LARA_27092026, RECIPES["heavy"]) ----
MAX_EPOCHS = 120
LR = 2e-4
WD = 1e-5
CLIP = 1.0
BATCH = 64
WORKERS = 2
PLATEAU_FACTOR = 0.5
PLATEAU_PATIENCE = 5
PLATEAU_MIN_LR = 1e-6
EARLY_PATIENCE = 8
SEED = 42

# ---- auxiliary-loss weights ----
# FIXED across CR
LAMBDA_STFT = 0.5
LAMBDA_WT = 0.2
LAMBDA_NCC = 0.3          # was an unrecorded make_loss default; now explicit
# SCALED linearly with CR: eff = base * (1 + CR_SCALE_K * (cr - min)/(max - min))
CR_SCALE_K = 2.0
BASE_LAMBDA_PHASE = 0.3   # -> 0.9 at CR=100
BASE_ARRIVAL_FLOOR = 1.0  # -> 3.0 at CR=100   (see docstring: this DILUTES)
STA_LEN = 20
LTA_LEN = 100

# ---- guards ----
EXPECTED_LARA_SHA = "ea4c8f4b8a2caaa3"   # see docstring
CR_TOL = 0.001                           # relative tolerance on the achieved CR
GATE_TOL_DB = 0.05                       # MSE reproduction gate

SMOKE = False


# =============================================================== device
def resolve_device(override=None):
    """CUDA -> XPU -> MPS -> CPU, so the script runs on whatever the box has."""
    if override:
        return torch.device(override)
    if torch.cuda.is_available():
        return torch.device("cuda")
    xpu = getattr(torch, "xpu", None)
    if xpu is not None and xpu.is_available():
        return torch.device("xpu")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def print_device_report(device):
    print(f"  device          : {device}")
    print(f"  torch           : {torch.__version__}")
    if device.type == "cuda":
        i = torch.cuda.current_device()
        cap = torch.cuda.get_device_capability(i)
        print(f"  GPU             : {torch.cuda.get_device_name(i)}")
        print(f"  capability      : {cap[0]}.{cap[1]}")
        print(f"  cuda            : {torch.version.cuda}")
        print(f"  device memory   : {torch.cuda.get_device_properties(i).total_memory / 2**30:.1f} GB")
    print(f"  cudnn benchmark : {torch.backends.cudnn.benchmark}")
    print(f"  cudnn TF32      : {torch.backends.cuda.matmul.allow_tf32}")
    print(f"  AMP             : disabled (changes numerics; off in the main run too)")


def set_seed(seed):
    """Full determinism, including cudnn. The predecessor study did the same."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =============================================================== data
class EfficientSeismicDataset(Dataset):
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        waveform = self.data[idx]
        z_min = np.min(waveform)
        z_max = np.max(waveform)
        
        if z_max - z_min > 0:
            waveform = (waveform - z_min) / (z_max - z_min)
        else:
            waveform = np.zeros_like(waveform)
        
        return torch.tensor(waveform, dtype=torch.float32).unsqueeze(0)


def load_filtered_stead(csv_path, h5_path, max_samples=None, component=COMPONENT):
    """Load the STEAD traces that pass the study filter, one component only.

    Filter: local earthquakes with source_magnitude > MIN_MAGNITUDE and
    source_distance_km <= MAX_DISTANCE_KM. Traces shorter than 1500 samples are
    skipped. Those two thresholds come from the CONFIG cell.
    """
    df = pd.read_csv(csv_path, low_memory=False,
                     usecols=["trace_name", "trace_category", "source_distance_km", "source_magnitude"])
    df = df[(df.trace_category == "earthquake_local") &
            (df.source_magnitude > MIN_MAGNITUDE) &
            (df.source_distance_km <= MAX_DISTANCE_KM)]
    names = df["trace_name"].to_list()
    if max_samples is not None:
        names = names[:max_samples]

    print(f"[data] filter: earthquake_local, magnitude > {MIN_MAGNITUDE}, "
          f"distance_km <= {MAX_DISTANCE_KM} -> pool = {len(df)}")
    print(f"[data] loading {len(names)} traces (component {component})")

    data = []
    with h5py.File(h5_path, "r") as f:
        for i, name in enumerate(names):
            try:
                w = f["data"][name][:]
                if w.shape[0] >= 1500:
                    data.append(w[:1500, component])
            except KeyError:
                continue
            if (i + 1) % 2000 == 0:
                print(f"  loaded {i + 1}/{len(names)}")

    print(f"[data] loaded waveforms: {len(data)}")
    return np.array(data)


# =============================================================== blocks
# VERBATIM from cell 5 of Final_LARA_27092026.ipynb. class LARA depends on
# both of these, so they are reproduced exactly as trained.
class AttentionBlock(nn.Module):
    """Simplified channel attention mechanism - EXACT ABLATION VERSION"""
    def __init__(self, channels):
        super().__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, max(4, channels//8), kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(max(4, channels//8), channels, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        attention_weights = self.attention(x)
        return x * attention_weights

class EnhancedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, use_attention=False):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.elu = nn.ELU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        
        self.attention = AttentionBlock(out_channels) if use_attention else nn.Identity()
            
    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.elu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.elu(out)
        out = self.attention(out)
        return out


# =============================================================== the model
# VERBATIM from cell 6 of Final_LARA_27092026.ipynb. Do not edit: verify_lara_hash()
# below checks this text against EXPECTED_LARA_SHA and aborts on any change.
class LARA(nn.Module):
    """
    LARA - Light Autoencoder for Reconstruction & Analysis.

    Single production configuration: variant='pyramid_funnel32' (see VARIANT in the CONFIG
    cell). Deep residual + channel-attention encoder, filled projection funnel, learned
    channel-pyramid decoder, conv refinement head.

    variant='pyramid_funnel32' (LARA): FILLS THE FUNNEL - T = L (no min(latent, L) clamp),
                               so the projection width no longer collapses as CR grows, and
                               BOTTLE_CHANNELS=32 -> 6016 projection columns (cr<=30) /
                               3008 (cr>30). cr<10 is identical to 'pyramid'.
    Compression ratio is preserved exactly: latent == input_len // cr (asserted).
    MSE loss only -> fair against the verbatim baseline recipe.

    Other internal tokens kept for provenance / ablations:
    variant='pyramid'   : single latent path; channel pyramid (16->32->64) + refinement head.
    variant='pyramid_v1': exact original 'pyramid' behaviour (archival alias).
    variant='pyramid_b32'    : 32-channel bottleneck spine for cr>=10 (16 for cr<10).
    variant='pyramid_b32_hd' : b32 + high-CR decoder polish for cr>=30 (ceiling 96, k9).
    variant='multiscale': latent split coarse (ceil half) + detail (rest), finest feature
                          carried in its own code, both decoded and fused.
    variant='pyramid_funnel64': same T = L fix with B=64 (cr<=30) / 128 (cr>30) -> 12032
                               columns at EVERY CR (baseline-parity width).
    """

    BOTTLE_CHANNELS = 16

    def __init__(self, cr, input_len=1500, variant="pyramid"):
        super().__init__()
        self.cr = cr
        self.input_len = input_len
        self.variant = variant
        self.latent_dim = input_len // cr

        # ---- variant knobs ----
        b32 = variant in ("pyramid_b32", "pyramid_b32_hd") and cr >= 10
        hd = variant == "pyramid_b32_hd" and cr >= 30
        funnel = variant in ("pyramid_funnel32", "pyramid_funnel64") and cr >= 10
        if variant == "pyramid_funnel32" and funnel:
            self.BOTTLE_CHANNELS = 32
        elif variant == "pyramid_funnel64" and funnel:
            self.BOTTLE_CHANNELS = 64 if cr <= 30 else 128
        else:
            self.BOTTLE_CHANNELS = 32 if b32 else 16
        self.pyramid_ceiling = max(64, self.BOTTLE_CHANNELS) if funnel else (96 if hd else 64)
        self.final_kernel = 9 if hd else 5

        self.L = 188 if cr <= 30 else 94
        self.T = self.L if funnel else min(self.latent_dim, self.L)

        if variant == "multiscale":
            self.latent_coarse = (self.latent_dim + 1) // 2
            self.latent_detail = self.latent_dim - self.latent_coarse
            self.T2 = min(32, max(2, self.latent_dim // 8))
        else:
            self.latent_coarse = self.latent_dim
            self.latent_detail = 0
            self.T2 = None
        assert self.latent_coarse + self.latent_detail == self.latent_dim, "CR must stay exact"

        # ---- encoder stages ----
        self.stage0 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(32),
            nn.ELU(inplace=True),
            AttentionBlock(32),
        )
        self.stage1 = EnhancedResidualBlock(32, 64, stride=2, use_attention=True)
        self.stage2 = EnhancedResidualBlock(64, 128, stride=2, use_attention=True)
        self.stage3 = EnhancedResidualBlock(128, 256, stride=2, use_attention=True) if cr > 30 else None
        final_channels = 256 if self.stage3 is not None else 128

        # ---- coarse bottleneck (original Light path) ----
        self.bot_conv = nn.Sequential(
            nn.Conv1d(final_channels, self.BOTTLE_CHANNELS, kernel_size=1),
            nn.ELU(inplace=True),
        )
        self.bot_pool = nn.Sequential(
            nn.AdaptiveAvgPool1d(self.T),
            nn.Flatten(),
        )
        self.coarse_head = nn.Linear(self.BOTTLE_CHANNELS * self.T, self.latent_coarse)

        # ---- detail bottleneck (multiscale only): finest 32ch @ 750 feature ----
        self.detail_block = nn.Sequential()
        self.detail_head = nn.Identity()
        if variant == "multiscale":
            self.detail_block = nn.Sequential(
                nn.Conv1d(32, 32, kernel_size=3, padding=1),
                nn.BatchNorm1d(32),
                nn.ELU(inplace=True),
                nn.AdaptiveAvgPool1d(self.T2),
                nn.Flatten(),
            )
            self.detail_head = nn.Linear(32 * self.T2, self.latent_detail)

        # ---- decoders (channel pyramid + learned upsampling) ----
        self.coarse_decoder = nn.Sequential(
            nn.Linear(self.latent_coarse, self.BOTTLE_CHANNELS * self.T),
            nn.Unflatten(1, (self.BOTTLE_CHANNELS, self.T)),
            nn.ELU(inplace=True),
            self._pyramid(self.BOTTLE_CHANNELS, self.T),
        )
        self.detail_decoder = nn.Sequential()
        if variant == "multiscale":
            self.detail_decoder = nn.Sequential(
                nn.Linear(self.latent_detail, self.BOTTLE_CHANNELS * self.T2),
                nn.Unflatten(1, (self.BOTTLE_CHANNELS, self.T2)),
                nn.ELU(inplace=True),
                self._pyramid(self.BOTTLE_CHANNELS, self.T2),
            )

        # ---- refinement head ----
        in_ch = 128 if variant == "multiscale" else self.pyramid_ceiling
        self.head = nn.Sequential(
            nn.Conv1d(in_ch, 32, kernel_size=5, padding=2),
            nn.ELU(inplace=True),
            nn.Conv1d(32, 1, kernel_size=15, padding=7),
            nn.Sigmoid(),
        )

        self.test_dimensions()

    # ------------------------------------------------------------------ #
    # encoding / decoding
    # ------------------------------------------------------------------ #
    def encoder(self, x):
        """Concatenated latent (coarse + detail) - the actual transmitted code."""
        zc = self.encode_coarse(x)
        if self.variant == "multiscale":
            zd = self.encode_detail(x)
            return torch.cat([zc, zd], dim=-1)
        return zc

    def encode_coarse(self, x):
        h = self.stage0(x)
        h = self.stage1(h)
        h = self.stage2(h)
        if self.stage3 is not None:
            h = self.stage3(h)
        f = self.bot_conv(h)
        f = self.bot_pool(f)
        return self.coarse_head(f)

    def encode_detail(self, x):
        f = self.stage0(x)
        d = self.detail_block(f)
        return self.detail_head(d)

    def forward(self, x):
        zc = self.encode_coarse(x)
        cf = self.coarse_decoder(zc)
        if self.variant == "multiscale":
            zd = self.encode_detail(x)
            df = self.detail_decoder(zd)
            fused = torch.cat([cf, df], dim=1)
            return self.head(fused)
        return self.head(cf)

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    def _pyramid(self, start_channels, start_len):
        """Channel-pyramid upsampler from start_len to input_len using learned
        (nearest + conv) upsampling. Channel ramp up to the variant ceiling."""
        nlev = 6
        ratio = self.input_len / start_len
        levels = []
        cch = start_channels
        prev_len = start_len
        for k in range(nlev):
            tgt = int(round(start_len * (ratio ** ((k + 1) / nlev))))
            tgt = max(tgt, prev_len + 1)
            if k == nlev - 1:
                tgt = self.input_len
            ksize = self.final_kernel if k == nlev - 1 else 5
            nc = min(self.pyramid_ceiling, int(cch * 2))
            levels.append(nn.Sequential(
                nn.Upsample(size=tgt, mode='nearest'),
                nn.Conv1d(cch, nc, kernel_size=ksize, padding=ksize // 2, bias=False),
                nn.BatchNorm1d(nc),
                nn.ELU(inplace=True),
            ))
            cch = nc
            prev_len = tgt
        return nn.Sequential(*levels)

    def test_dimensions(self):
        try:
            with torch.no_grad():
                test_input = torch.randn(1, 1, self.input_len)
                zc = self.encode_coarse(test_input)
                assert zc.numel() == self.latent_coarse, f"coarse latent {zc.numel()} != {self.latent_coarse}"
                if self.variant == "multiscale":
                    zd = self.encode_detail(test_input)
                    assert zd.numel() == self.latent_detail, f"detail latent {zd.numel()} != {self.latent_detail}"
                total = self.encoder(test_input).numel()
                assert total == self.latent_dim, f"total latent {total} != {self.latent_dim}"
                decoded = self(test_input)
                assert decoded.shape[2] == self.input_len, f"output {decoded.shape[2]} != {self.input_len}"
                actual_cr = self.input_len / total
                print(f"[OK] Light-HARA-V2 ({self.variant}) CR={self.cr}: Input {self.input_len} -> Latent {total} "
                      f"(coarse {self.latent_coarse} + detail {self.latent_detail}) -> Output {decoded.shape[2]} "
                      f"(True CR = {actual_cr:.2f})")
        except Exception as e:
            print(f"[FAIL] Light-HARA-V2 ({self.variant}) CR={self.cr}: {e}")
            self.create_fallback_architecture()

    def create_fallback_architecture(self):
        print("Creating fallback architecture...")
        self.stage0 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=15, stride=2, padding=7), nn.ELU(),
            nn.Conv1d(32, 64, kernel_size=9, stride=2, padding=4), nn.ELU(),
        )
        self.stage1 = nn.Identity()
        self.stage2 = nn.Identity()
        self.stage3 = None
        self.bot_conv = nn.Conv1d(64, self.BOTTLE_CHANNELS, kernel_size=1)
        self.bot_pool = nn.Sequential(nn.AdaptiveAvgPool1d(self.T), nn.Flatten())
        self.coarse_head = nn.Linear(self.BOTTLE_CHANNELS * self.T, self.latent_dim)
        self.latent_coarse = self.latent_dim
        self.latent_detail = 0
        self.variant = "pyramid"
        self.detail_block = nn.Sequential()
        self.detail_head = nn.Identity()
        self.coarse_decoder = nn.Sequential(
            nn.Linear(self.latent_dim, self.BOTTLE_CHANNELS * self.T),
            nn.Unflatten(1, (self.BOTTLE_CHANNELS, self.T)), nn.ELU(),
            nn.Upsample(size=self.input_len, mode='linear', align_corners=False),
            nn.Conv1d(self.BOTTLE_CHANNELS, 1, kernel_size=15, padding=7), nn.Sigmoid(),
        )
        self.detail_decoder = nn.Sequential()
        self.head = nn.Identity()
        print("[OK] Fallback architecture created")


# =============================================================== guards
def _slice_embedded_class_lara():
    """Read this file and slice out the class LARA text exactly as the assembler did.

    Reading __file__ is used in preference to inspect.getsource(), which fails
    when this module is loaded through importlib rather than run as a script.
    Slicing the source text is deterministic and matches the extraction rule
    used to embed the class in the first place.
    """
    text = open(os.path.abspath(__file__), encoding="utf-8").read()
    marker = "class LARA(nn.Module):"
    i = text.index(marker)
    cands = [c for c in (text.find("\nclass ", i + 10), text.find("\n# =", i + 10)) if c > 0]
    j = min(cands) if cands else len(text)
    return text[i:j].rstrip()


def verify_lara_hash(from_notebook=None):
    """Refuse to run against anything but the topology that was actually trained."""
    text = _slice_embedded_class_lara()
    if not text.startswith("class LARA(nn.Module):"):
        raise SystemExit("ABORT: could not locate class LARA in this file")
    sha = hashlib.sha256(text.encode()).hexdigest()
    ok = sha.startswith(EXPECTED_LARA_SHA)
    print(f"[guard] class LARA sha256 {sha[:16]} -> {'MATCH' if ok else 'MISMATCH'}")
    if not ok:
        raise SystemExit(
            f"ABORT: class LARA hashes to {sha[:16]}, expected {EXPECTED_LARA_SHA}.\n"
            "       The embedded architecture no longer matches the trained model."
        )

    if from_notebook:
        if not os.path.exists(from_notebook):
            print(f"[guard] notebook cross-check skipped: {from_notebook} not found")
            return sha
        cells = {}
        nb = json.load(open(from_notebook, encoding="utf-8"))
        for i, c in enumerate(nb["cells"]):
            if c["cell_type"] == "code":
                cells[i] = "".join(c["source"])
        src_cell = next(s for s in cells.values() if "class LARA(nn.Module)" in s)
        i0 = src_cell.index("class LARA(nn.Module)")
        i1 = src_cell.find("\nclass ", i0 + 10)
        nb_text = (src_cell[i0:i1] if i1 > 0 else src_cell[i0:]).rstrip()
        nb_sha = hashlib.sha256(nb_text.encode()).hexdigest()
        print(f"[guard] notebook copy  sha256 {nb_sha[:16]} -> "
              f"{'MATCH' if nb_sha == sha else 'MISMATCH'}  ({from_notebook})")
        if nb_sha != sha:
            raise SystemExit(
                "ABORT: the embedded class LARA and the notebook copy differ.\n"
                f"       embedded {sha[:16]} vs notebook {nb_sha[:16]}"
            )
    return sha


def measure_latent(model, input_len=INPUT_LEN, device=None):
    """Measured latent width. Never assumes the requested CR."""
    dev = device or next(model.parameters()).device
    probe = torch.randn(1, 1, input_len, device=dev)
    if hasattr(model, "encoder"):
        with torch.no_grad():
            return int(model.encoder(probe).numel()), "encoder method"
    captured = {}
    handle = model.decoder.register_forward_pre_hook(
        lambda m, inp: captured.__setitem__("z", int(inp[0].shape[1]))
    )
    try:
        with torch.no_grad():
            model(probe)
    finally:
        handle.remove()
    return captured["z"], "decoder hook"


def preflight_cr(device):
    """Build LARA at every CR and MEASURE the latent, before spending any GPU time.

    A dimension bug surfaces in seconds here instead of after eight hours.
    """
    print("\nCR pre-flight")
    print("=" * 92)
    print(f"{'CR':>4} | {'latent':>7} | {'expected':>8} | {'achieved CR':>11} | "
          f"{'rel err':>9} | {'params':>10} | {'probe':<14} | verdict")
    print("-" * 92)
    rows, failures = [], []
    for cr in CRS:
        try:
            model = LARA(cr, variant=LARA_VARIANT)
        except Exception as e:
            failures.append((cr, -1, f"build failed: {e}"))
            print(f"{cr:>4} | build failed: {e}")
            continue
        model = model.to(device)
        latent, how = measure_latent(model, device=device)
        expected = INPUT_LEN // cr
        achieved = INPUT_LEN / latent
        rel = abs(achieved - cr) / cr
        ok = (rel <= CR_TOL) and (latent == expected)
        nparam = sum(p.numel() for p in model.parameters())
        rows.append((cr, latent, achieved, nparam))
        if not ok:
            failures.append((cr, latent, f"rel err {100 * rel:.4f}%"))
        print(f"{cr:>4} | {latent:>7} | {expected:>8} | {achieved:>11.4f} | "
              f"{100 * rel:>8.4f}% | {nparam:>10,} | {how:<14} | "
              f"{'PASS' if ok else 'FAIL'}")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print("-" * 92)
    if rows:
        print(f"{len(rows)} combinations, worst relative error "
              f"{100 * max(abs(r[2] - c) / c for c, r in zip(CRS, rows)):.6f}%, "
              f"tolerance {100 * CR_TOL:.1f}%")
    off_grid = [c for c in CRS if INPUT_LEN % c]
    print(f"CRs dividing {INPUT_LEN} evenly: {'all' if not off_grid else off_grid}")
    if failures:
        detail = "\n".join(f"  CR={f[0]}: {f[2]}" for f in failures)
        raise RuntimeError(f"CR pre-flight FAILED ({len(failures)} of {len(rows)}):\n{detail}")
    print("CR pre-flight: PASS")


# =============================================================== metric helpers
def calculate_psnr(original, reconstructed):
    """Data range is 1.0 because the traces are min-max scaled to [0, 1]."""
    mse = float(np.mean((original - reconstructed) ** 2))
    if mse == 0:
        return float("inf")
    return 20 * math.log10(1.0 / math.sqrt(mse))


def calculate_ssim_1d(original, reconstructed, data_range=1.0):
    if original.shape != reconstructed.shape:
        n = min(original.shape[-1], reconstructed.shape[-1])
        original, reconstructed = original[:n], reconstructed[:n]
    return ssim(np.asarray(original).ravel(), np.asarray(reconstructed).ravel(),
                data_range=data_range)


def calculate_model_size(model):
    total = 0
    for p in model.parameters():
        total += p.nelement() * p.element_size()
    for b in model.buffers():
        total += b.nelement() * b.element_size()
    return total / 1024 ** 2


# =============================================================== losses
# VERBATIM from cell 3 of loss_ablation_light_GPU.ipynb, except for the two
# documented fixes above. See the module docstring for the defects that are
# preserved on purpose.
def stft_spectral_loss(x, x_hat, n_fft_list=(128, 256, 512), hop_ratio=0.25, log_w=0.05):
    """MSE on STFT magnitudes across several window sizes + a log-magnitude L1 term.

    NOTE: this term only compares |STFT|. It is invariant to absolute phase,
    so on its own it cannot penalize a polarity-inverted reconstruction.
    Pair it with stft_phase_loss / ncc_loss if phase or first-motion polarity
    matters (it does, for P-wave first-motion analysis).

    x, x_hat: (batch, 1, length)
    """
    signal = x.squeeze(1)
    signal_hat = x_hat.squeeze(1)
    total = 0.0
    for n in n_fft_list:
        hop = int(n * hop_ratio)
        window = torch.hann_window(n, device=x.device)
        sx = torch.stft(signal, n_fft=n, hop_length=hop, window=window, return_complex=True)
        sy = torch.stft(signal_hat, n_fft=n, hop_length=hop, window=window, return_complex=True)
        mx, my = sx.abs(), sy.abs()
        total = total + F.mse_loss(mx, my)
        total = total + log_w * F.l1_loss(torch.log1p(mx), torch.log1p(my))
    return total / len(n_fft_list)


def _db4_filters(device, wavelet='db4'):
    w = pywt.Wavelet(wavelet)
    lo = torch.tensor(w.dec_lo, dtype=torch.float32, device=device).reshape(1, 1, -1)
    hi = torch.tensor(w.dec_hi, dtype=torch.float32, device=device).reshape(1, 1, -1)
    return lo, hi


class TorchWaveletDecompose:
    """Multi-level Mallat decomposition with fixed analysis filters (differentiable)."""

    def __init__(self, wavelet='db4', level=3, device='cpu'):
        self.level = level
        self.lo, self.hi = _db4_filters(device, wavelet)
        self.pad = self.lo.shape[-1] // 2

    def decompose(self, x):
        # x: (batch, 1, length) -> list of coeff sets: details per level + final approx
        coeffs = []
        c = x
        for _ in range(self.level):
            c_a = F.conv1d(c, self.lo, stride=2, padding=self.pad)
            c_d = F.conv1d(c, self.hi, stride=2, padding=self.pad)
            coeffs.append(c_d)
            c = c_a
        coeffs.append(c)
        return coeffs


def torch_wavelet_loss(x, x_hat, level=3, wavelet='db4', detail_weight=1.0, approx_weight=0.5):
    """Size-normalized L1 on wavelet coefficients, details weighted above approximation.

    Good for transient, non-stationary events like P/S onsets: unlike fixed-window
    STFT, wavelets give better time localization at high frequencies, which is
    where onset sharpness lives.
    """
    decomp = TorchWaveletDecompose(wavelet=wavelet, level=level, device=x.device)
    dx = decomp.decompose(x)
    dy = decomp.decompose(x_hat)
    total = 0.0
    for i in range(level):
        total = total + detail_weight * F.l1_loss(dx[i], dy[i])
    total = total + approx_weight * F.l1_loss(dx[-1], dy[-1])
    return total


def stft_phase_loss(x, x_hat, n_fft_list=(128, 256, 512), hop_ratio=0.25, eps=1e-8):
    """Penalizes phase-derivative (instantaneous frequency) mismatch.

    This is what catches things magnitude-only STFT loss cannot: e.g. a
    reconstruction with correct spectral magnitude but flipped or shifted
    phase (including a polarity-inverted P arrival) will score badly here
    even though stft_spectral_loss would be near zero.
    """
    signal = x.squeeze(1)
    signal_hat = x_hat.squeeze(1)
    total = 0.0
    for n in n_fft_list:
        hop = int(n * hop_ratio)
        window = torch.hann_window(n, device=x.device)
        sx = torch.stft(signal, n_fft=n, hop_length=hop, window=window, return_complex=True)
        sy = torch.stft(signal_hat, n_fft=n, hop_length=hop, window=window, return_complex=True)
        # unit phasors (magnitude divided out) -> pure phase information
        px = sx / (sx.abs() + eps)
        py = sy / (sy.abs() + eps)
        # frame-to-frame phase difference, represented as a phasor so there's
        # no atan2 / unwrapping involved
        dpx = px[..., 1:] * torch.conj(px[..., :-1])
        dpy = py[..., 1:] * torch.conj(py[..., :-1])
        total = total + F.mse_loss(torch.view_as_real(dpx), torch.view_as_real(dpy))
    return total / len(n_fft_list)


def _hilbert_analytic(x, eps=1e-8):
    # x: (batch, 1, length) real -> analytic signal via FFT-domain Hilbert transform
    n = x.shape[-1]
    xf = torch.fft.fft(x, dim=-1)
    h = torch.zeros(n, device=x.device, dtype=xf.dtype)
    if n % 2 == 0:
        h[0] = 1
        h[n // 2] = 1
        h[1:n // 2] = 2
    else:
        h[0] = 1
        h[1:(n + 1) // 2] = 2
    analytic = torch.fft.ifft(xf * h, dim=-1)
    return analytic


def envelope_phase_loss(x, x_hat, envelope_weight=1.0, phase_weight=0.5, eps=1e-8):
    ax = _hilbert_analytic(x, eps)
    ay = _hilbert_analytic(x_hat, eps)
    env_x, env_y = ax.abs(), ay.abs()
    env_loss = F.l1_loss(env_x, env_y)
    # unit phasors, compared directly (avoids unwrapping, stays smooth)
    phx = ax / (env_x + eps)
    phy = ay / (env_y + eps)
    phase_loss = F.mse_loss(torch.view_as_real(phx), torch.view_as_real(phy))
    return envelope_weight * env_loss + phase_weight * phase_loss


def ncc_loss(x, x_hat, eps=1e-8):
    signal = x.squeeze(1)
    signal_hat = x_hat.squeeze(1)
    xm = signal - signal.mean(dim=-1, keepdim=True)
    ym = signal_hat - signal_hat.mean(dim=-1, keepdim=True)
    num = (xm * ym).sum(dim=-1)
    den = torch.sqrt((xm ** 2).sum(dim=-1) * (ym ** 2).sum(dim=-1) + eps)
    ncc = num / den  # in [-1, 1]; 1 = perfect match, -1 = perfectly inverted
    return (1.0 - ncc).mean()


def stalta_weights(x, sta_len=20, lta_len=100, floor=1.0, eps=1e-8):
    with torch.no_grad():
        energy = x.pow(2)
        sta = F.avg_pool1d(energy, kernel_size=sta_len, stride=1, padding=sta_len // 2)
        lta = F.avg_pool1d(energy, kernel_size=lta_len, stride=1, padding=lta_len // 2)
        ratio = sta / (lta + eps)
        ratio = ratio[..., :x.shape[-1]]  # crop any off-by-one from even-kernel padding
        peak = ratio.amax(dim=-1, keepdim=True) + eps
        w = floor + ratio / peak
    return w


def arrival_weighted_mse(x, x_hat, sta_len=20, lta_len=100, floor=1.0):
    w = stalta_weights(x, sta_len=sta_len, lta_len=lta_len, floor=floor)
    return (w * (x - x_hat) ** 2).mean()


def make_loss(mode, lambda_stft=0.5, lambda_wt=0.2, lambda_phase=0.3,
              lambda_env=0.3, lambda_ncc=LAMBDA_NCC,
              sta_len=20, lta_len=100, arrival_floor=1.0):
    def loss_fn(x, x_hat):
        if 'arrival' in mode:
            total = arrival_weighted_mse(x, x_hat, sta_len=sta_len, lta_len=lta_len,
                                          floor=arrival_floor)
        else:
            total = F.mse_loss(x_hat, x)
        if 'stft' in mode:
            total = total + lambda_stft * stft_spectral_loss(x, x_hat)
        if 'wt' in mode:
            total = total + lambda_wt * torch_wavelet_loss(x, x_hat)
        if 'phase' in mode:
            total = total + lambda_phase * stft_phase_loss(x, x_hat)
        if 'env' in mode:
            total = total + lambda_env * envelope_phase_loss(x, x_hat)
        if 'ncc' in mode:
            total = total + lambda_ncc * ncc_loss(x, x_hat)
        return total
    return loss_fn


# =============================================================== arm wiring
LARA_VARIANT = "pyramid_funnel32"

# Order matches the predecessor study so the two are directly comparable.
# (name, make_loss mode, lambda_stft, lambda_wt)
ARMS = [
    ("MSE",                "mse",                0.0, 0.0),   # control
    ("STFT_Arrival",       "stft_arrival",       0.5, 0.0),
    ("STFT_Phase",         "stft_phase",         0.5, 0.0),
    ("STFT_Wavelet",       "stft_wt",            0.5, 0.2),
    ("STFT_Phase_Arrival", "stft_phase_arrival", 0.5, 0.0),
    ("STFT",               "stft",               0.5, 0.0),
    ("NCC_Arrival",        "ncc_arrival",        0.0, 0.0),
]


def cr_scaled_lambda(base_lambda, cr, crs=None, k=CR_SCALE_K):
    """base -> base * (1 + k) at max CR. Phase and arrival terms only."""
    crs = crs or CRS
    if base_lambda == 0.0:
        return 0.0
    lo, hi = min(crs), max(crs)
    if hi == lo:
        return base_lambda
    return base_lambda * (1.0 + k * (cr - lo) / (hi - lo))


def build_loss(mode, cr):
    """Effective lambdas for one (mode, cr). Only phase and arrival scale."""
    lam_stft = LAMBDA_STFT if "stft" in mode else 0.0
    lam_wt = LAMBDA_WT if "wt" in mode else 0.0
    lam_phase = cr_scaled_lambda(BASE_LAMBDA_PHASE, cr) if "phase" in mode else 0.0
    floor = cr_scaled_lambda(BASE_ARRIVAL_FLOOR, cr) if "arrival" in mode else 1.0
    fn = make_loss(
        mode,
        lambda_stft=lam_stft,
        lambda_wt=lam_wt,
        lambda_phase=lam_phase,
        lambda_ncc=LAMBDA_NCC,
        arrival_floor=floor,
        sta_len=STA_LEN,
        lta_len=LTA_LEN,
    )
    return fn, lam_stft, lam_wt, lam_phase, floor


# =============================================================== training
def train_model(model, train_loader, val_loader, loss_fn, run_tag, device, ckpt_dir,
                max_epochs=MAX_EPOCHS):
    """Uniform budget. Early stopping decides the effective duration.

    Identical recipe to Final_LARA_27092026: Adam, ReduceLROnPlateau(factor 0.5,
    patience 5, min_lr 1e-6), gradient clipping, early stopping at patience 8,
    120-epoch cap. Validation runs the SAME loss as training.
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=PLATEAU_FACTOR, patience=PLATEAU_PATIENCE,
        min_lr=PLATEAU_MIN_LR,
    )
    ckpt = os.path.join(ckpt_dir, f"best_LARA_{run_tag}.pth")

    best_val = float("inf")
    patience_counter = 0
    best_state = copy.deepcopy(model.state_dict())
    epochs_used = 0
    t0 = time.time()

    for epoch in range(max_epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(batch, model(batch))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            optimizer.step()
            train_loss += float(loss.item()) * batch.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                val_loss += float(loss_fn(batch, model(batch)).item()) * batch.size(0)
        val_loss /= len(val_loader.dataset)

        scheduler.step(val_loss)

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            patience_counter = 0
            best_state = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), ckpt)
        else:
            patience_counter += 1
            if patience_counter >= EARLY_PATIENCE:
                epochs_used = epoch + 1
                break
        epochs_used = epoch + 1

        if epoch % 5 == 0:
            print(f"      epoch {epoch + 1:>3}/{max_epochs}  "
                  f"train {train_loss:.6f}  val {val_loss:.6f}  "
                  f"lr {optimizer.param_groups[0]['lr']:.2e}", flush=True)

    model.load_state_dict(best_state)
    dt = (time.time() - t0) / 60.0
    print(f"      -> best val {best_val:.6f} in {epochs_used} epochs, {dt:.1f} min, "
          f"ckpt {os.path.basename(ckpt)}", flush=True)
    return model, epochs_used


# =============================================================== evaluation
def eval_all(model, test_data_np, device):
    """Per-trace metrics, then averaged. Matches the main run's evaluate_model.

    PolarityMatch and PickErr_samples are deliberately absent; the module
    docstring explains why both are invalid under this preprocessing.
    """
    model.eval()
    with torch.no_grad():
        x = torch.tensor(test_data_np, dtype=torch.float32)
        if x.ndim == 2:
            x = x.unsqueeze(1)
        rec = model(x.to(device)).cpu().numpy()
    orig = test_data_np
    if rec.ndim == 3:
        rec = rec[:, 0, :]

    snr, psnr, ssim_v, mse_v, corr, mae = [], [], [], [], [], []
    for i in range(orig.shape[0]):
        o, p = orig[i], rec[i]
        n = min(len(o), len(p))
        o, p = o[:n], p[:n]
        m = float(np.mean((o - p) ** 2))
        sig = float(np.mean(o ** 2))
        snr.append(10 * math.log10(sig / (m + 1e-12)) if m > 0 else float("inf"))
        mse_v.append(m)
        psnr.append(calculate_psnr(o, p))
        ssim_v.append(calculate_ssim_1d(o, p))
        corr.append(float(np.corrcoef(o, p)[0, 1]) if (np.std(o) > 0 and np.std(p) > 0) else 0.0)
        mae.append(float(np.mean(np.abs(o - p))))

    snr = np.asarray(snr, dtype=float)
    finite = snr[np.isfinite(snr)]
    return {
        "SNR": float(np.mean(finite)) if finite.size else float("inf"),
        "SNR_std": float(np.std(finite)) if finite.size else 0.0,
        "SNR_median": float(np.median(finite)) if finite.size else float("inf"),
        "PSNR": float(np.mean(psnr)),
        "SSIM": float(np.mean(ssim_v)),
        "MSE": float(np.mean(mse_v)),
        "Correlation": float(np.mean(corr)),
        "MAE": float(np.mean(mae)),
    }


# =============================================================== main
def main():
    global CRS, SMOKE, N_TRACES, MAX_EPOCHS

    ap = argparse.ArgumentParser(
        description="Loss-function ablation on LARA (answers Reviewer 2, Comment 4)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--csv", default=CSV_PATH, help="STEAD merged.csv")
    ap.add_argument("--h5", default=H5_PATH, help="STEAD merged.hdf5")
    ap.add_argument("--out", default=OUT_DIR, help="output directory")
    ap.add_argument("--cr", default=None, help="comma-separated CRs, e.g. 2,20,100")
    ap.add_argument("--arms", default=None, help="comma-separated arm names")
    ap.add_argument("--n-traces", type=int, default=N_TRACES, help="None = all filtered")
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--workers", type=int, default=WORKERS)
    ap.add_argument("--max-epochs", type=int, default=None,
                    help="epoch cap; defaults to 120, or 2 under --smoke")
    ap.add_argument("--device", default=None, help="cuda | cuda:0 | xpu | mps | cpu")
    ap.add_argument("--smoke", action="store_true",
                    help="100 traces, 1 CR, 2 epochs, all 7 arms (~2 min)")
    ap.add_argument("--strict", action="store_true",
                    help="abort if the MSE arm does not reproduce the published SNR")
    ap.add_argument("--no-gate", action="store_true", help="skip the reproduction check")
    ap.add_argument("--from-notebook", default=SOURCE_NOTEBOOK,
                    help="cross-check the embedded class LARA against this notebook "
                         "(skipped when the path does not exist)")
    ap.add_argument("--published-csv", default=PUBLISHED_CSV,
                    help="model_comparison_results.csv for the reproduction check")
    args = ap.parse_args()

    # --smoke supplies the small defaults; an explicit --n-traces / --cr /
    # --max-epochs on the command line always wins, so
    # `--smoke --n-traces 40` really does use 40 traces.
    if args.smoke:
        SMOKE = True
        N_TRACES = 100 if args.n_traces is None else args.n_traces
        CRS = [10] if args.cr is None else [int(c) for c in args.cr.split(",") if c.strip()]
        MAX_EPOCHS = 2 if args.max_epochs is None else args.max_epochs
    else:
        if args.n_traces is not None:
            N_TRACES = args.n_traces
        if args.max_epochs is not None:
            MAX_EPOCHS = args.max_epochs
    if args.cr:
        CRS = [int(c) for c in args.cr.split(",") if c.strip()]
    arms = ARMS
    if args.arms:
        want = {a.strip() for a in args.arms.split(",") if a.strip()}
        arms = [a for a in ARMS if a[0] in want]
        if len(arms) != len(want):
            raise SystemExit(f"ABORT: unknown arm name; known: {[a[0] for a in ARMS]}")

    out_dir = os.path.abspath(args.out)
    if SMOKE:
        out_dir = out_dir + "_smoke"
    ckpt_dir = os.path.join(out_dir, "ckpts")
    os.makedirs(ckpt_dir, exist_ok=True)

    device = resolve_device(args.device)

    print("=" * 78)
    print("Loss-function ablation on LARA  --  Reviewer 2, Comment 4")
    print("=" * 78)
    print(f"  variant          : {LARA_VARIANT}")
    print(f"  arms             : {[a[0] for a in arms]}")
    print(f"  CRs              : {CRS}")
    print(f"  runs             : {len(arms) * len(CRS)}  (all from scratch)")
    print(f"  filter           : earthquake_local, magnitude > {MIN_MAGNITUDE}, "
          f"distance <= {MAX_DISTANCE_KM} km")
    print(f"  component        : {COMPONENT}    N_TRACES={N_TRACES}    N_EVAL={N_EVAL}")
    print(f"  recipe           : Adam lr={LR} wd={WD} clip={CLIP} batch={args.batch} "
          f"max_epochs={MAX_EPOCHS} early_patience={EARLY_PATIENCE}")
    print(f"  lambdas          : stft={LAMBDA_STFT} wt={LAMBDA_WT} ncc={LAMBDA_NCC} "
          f"phase {BASE_LAMBDA_PHASE}->{cr_scaled_lambda(BASE_LAMBDA_PHASE, max(CRS)):.4f} "
          f"floor {BASE_ARRIVAL_FLOOR}->{cr_scaled_lambda(BASE_ARRIVAL_FLOOR, max(CRS)):.4f}")
    print(f"  output           : {out_dir}")
    print(f"  smoke            : {SMOKE}")
    print_device_report(device)
    print("=" * 78)

    if not SMOKE:
        verify_lara_hash(args.from_notebook)
        preflight_cr(device)

    # ---- data ----
    print("\n[1/5] loading data")
    waveforms = load_filtered_stead(args.csv, args.h5, max_samples=N_TRACES,
                                   component=COMPONENT)
    if len(waveforms) == 0:
        raise SystemExit(f"ABORT: no waveforms loaded from {args.h5}")
    print(f"  waveforms {waveforms.shape}")

    dataset = EfficientSeismicDataset(waveforms)
    n = len(dataset)
    test_size = int(0.1 * n)
    train_size = n - test_size - int(0.1 * n)
    train_val, test_dataset = train_test_split(dataset, test_size=test_size, random_state=SEED)
    train_dataset, val_dataset = train_test_split(train_val, train_size=train_size,
                                                  random_state=SEED)
    train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True,
                              num_workers=args.workers, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_dataset, batch_size=args.batch, shuffle=False,
                            num_workers=args.workers, pin_memory=(device.type == "cuda"))

    # Same evaluation subset as the main run: the FIRST N_EVAL traces of the test
    # split. Taking all of them would not reproduce model_comparison_results.csv.
    n_eval = min(N_EVAL, len(test_dataset))
    test_data = np.array([test_dataset[i] for i in range(n_eval)])
    test_data_np = test_data[:, 0, :]
    print(f"  split train={len(train_dataset)} val={len(val_dataset)} "
          f"test={len(test_dataset)}  eval subset={n_eval}")
    print(f"  test_data_np {test_data_np.shape}")

    # ---- run ----
    print("\n[2/5] training")
    rows = []
    total = len(arms) * len(CRS)
    done = 0
    t_all = time.time()
    for cr in CRS:
        for name, mode, _lam_stft, _lam_wt in arms:
            done += 1
            print(f"\n  [{done}/{total}] CR={cr}  arm={name}  mode={mode}", flush=True)
            # Identical initial weights across the seven arms at a given CR.
            set_seed(SEED)
            model = LARA(cr, variant=LARA_VARIANT)
            assert getattr(model, "variant", None) == LARA_VARIANT, \
                f"LARA fell back to the fallback architecture at CR={cr}"
            loss_fn, ls, lw, lp, fl = build_loss(mode, cr)
            print(f"      lambda_stft={ls:.4f} lambda_wt={lw:.4f} "
                  f"lambda_phase={lp:.4f} lambda_ncc={LAMBDA_NCC:.4f} "
                  f"arrival_floor={fl:.4f}  params={sum(p.numel() for p in model.parameters()):,}")
            model, epochs_used = train_model(
                model, train_loader, val_loader, loss_fn, f"cr{cr}_{name}", device, ckpt_dir,
                max_epochs=MAX_EPOCHS,
            )
            metrics = eval_all(model, test_data_np, device)
            latent, _ = measure_latent(model, device=device)
            metrics.update({
                "Model": "LARA",
                "Arm": name,
                "CR": cr,
                "Loss_Mode": mode,
                "Lambda_STFT": round(ls, 6),
                "Lambda_WT": round(lw, 6),
                "Lambda_NCC": round(LAMBDA_NCC, 6) if "ncc" in mode else 0.0,
                "Lambda_Phase_eff": round(lp, 6),
                "Arrival_Floor_eff": round(fl, 6),
                "Epochs_Used": epochs_used,
                "Parameter_Count": sum(p.numel() for p in model.parameters()),
                "Model_Size_MB": round(calculate_model_size(model), 4),
                "Latent_Dim": latent,
                "Actual_CR": round(INPUT_LEN / latent, 6),
            })
            rows.append(metrics)
            print(f"      SNR={metrics['SNR']:.3f}  SSIM={metrics['SSIM']:.4f}  "
                  f"corr={metrics['Correlation']:.4f}  MSE={metrics['MSE']:.6f}")
            del model, loss_fn
            if device.type == "cuda":
                torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    sel_cols = ["Model", "Arm", "CR", "Actual_CR", "Loss_Mode", "Lambda_STFT", "Lambda_WT",
                "Lambda_NCC", "Lambda_Phase_eff", "Arrival_Floor_eff", "Epochs_Used",
                "Parameter_Count", "Model_Size_MB", "Latent_Dim", "SNR", "SNR_std",
                "SNR_median", "PSNR", "SSIM", "MSE", "Correlation", "MAE"]
    sel_csv = os.path.join(out_dir, "loss_ablation_LARA_selection.csv")
    df[sel_cols].to_csv(sel_csv, index=False)
    print(f"\n[3/5] wrote {sel_csv}  ({len(df)} rows)")

    # ---- deltas, ranking ----
    print("\n[4/5] ranking")
    base_cols = ["CR", "SNR", "PSNR", "SSIM", "MSE", "Correlation"]
    base = df[df.Arm == "MSE"][base_cols].set_index("CR")
    rank_rows, percr_rows = [], []
    for name, grp in df.groupby("Arm"):
        if name == "MSE":
            continue
        if len(base) == 0:
            continue
        m = grp.merge(base, on="CR", suffixes=("", "_base"))
        for _, r in m.iterrows():
            percr_rows.append({
                "Arm": name, "CR": r.CR,
                "dSNR": r.SNR - r.SNR_base,
                "dPSNR": r.PSNR - r.PSNR_base,
                "dSSIM": r.SSIM - r.SSIM_base,
                "dMSE": r.MSE - r.MSE_base,
                "dCorrelation": r.Correlation - r.Correlation_base,
            })
        w = m.CR.apply(lambda c: 3.0 if c in HIGH_CR else 1.0).to_numpy()
        dsnr = m.SNR.to_numpy() - m.SNR_base.to_numpy()
        rank_rows.append({
            "Arm": name,
            "Mean_dSNR_dB": round(float(dsnr.mean()), 3),
            "Median_dSNR_dB": round(float(np.median(dsnr)), 3),
            "Max_dSNR_dB": round(float(dsnr.max()), 3),
            "Min_dSNR_dB": round(float(dsnr.min()), 3),
            "CRs_won_vs_MSE": int((dsnr > 0).sum()),
            "CRs_lost_vs_MSE": int((dsnr < 0).sum()),
            "Weighted_dSNR_dB_LEGACY": round(float(np.average(dsnr, weights=w)), 3),
            "CRs_with_SNR_regression_total": int((dsnr < 0).sum()),
            "CRs_with_SNR_regression_highCR": int(((dsnr < 0) & m.CR.isin(HIGH_CR)).sum()),
            "N": len(m),
        })

    if not rank_rows:
        print("  only the MSE control was run, so there is nothing to rank against it.\n"
              "  The reproduction check below still applies.")
    ranking = pd.DataFrame(rank_rows)
    if not ranking.empty:
        ranking = ranking.sort_values(
            ["Mean_dSNR_dB", "CRs_won_vs_MSE"],
            ascending=[False, False]).reset_index(drop=True)
        ranking.insert(0, "Rank_by_mean_dSNR", range(1, len(ranking) + 1))
        ranking["Rank_by_legacy_weighted"] = (
            ranking["Weighted_dSNR_dB_LEGACY"].rank(ascending=False, method="min").astype(int))
        print("\n  ranking (unweighted mean dSNR vs the MSE control)")
        print(ranking.to_string(index=False))
    rank_csv = os.path.join(out_dir, "loss_ablation_LARA_ranking.csv")
    ranking.to_csv(rank_csv, index=False)

    if percr_rows:
        percr_df = pd.DataFrame(percr_rows)
        percr_df.to_csv(os.path.join(out_dir, "loss_ablation_LARA_percr_deltas.csv"), index=False)
        percr_df.pivot(index="CR", columns="Arm", values="dSNR").round(4).to_csv(
            os.path.join(out_dir, "loss_ablation_LARA_percr_dSNR.csv"))

    if ranking.empty:
        n_high = 0
        reg = None
    else:
        reg = ranking["CRs_with_SNR_regression_highCR"]
        n_high = len(HIGH_CR.intersection(set(df.CR.astype(int))))
    if n_high == 0:
        print(f"\n  [note] no ratio from HIGH_CR {sorted(HIGH_CR)} is in this grid, so "
              "the high-CR regression column is trivially zero for every arm.")
    elif reg.nunique() == 1:
        print(f"\n  [note] CRs_with_SNR_regression_highCR is {int(reg.iloc[0])} for every "
              "arm, so it carries no information. The legacy 3x weighting puts "
              f"{200 * n_high // len(CRS)} % of the weight on {sorted(HIGH_CR)}; "
              "see the module docstring.")

    # ---- MSE reproduction check ----
    print("\n[5/5] figure + reproduction check")
    repro_rows = []
    if not args.no_gate and os.path.exists(args.published_csv):
        pub = pd.read_csv(args.published_csv)
        pub = pub[pub.Model == "LARA"]
        for _, r in df[df.Arm == "MSE"].iterrows():
            cr = int(r.CR)
            match = pub[pub.CR.astype(int) == cr]
            if match.empty:
                continue
            expected = float(match.SNR.iloc[0])
            delta = float(r.SNR) - expected
            repro_rows.append({
                "CR": cr, "Published_LARA_SNR_dB": round(expected, 4),
                "FromScratch_MSE_SNR_dB": round(float(r.SNR), 4),
                "Delta_dB": round(delta, 4),
                "Within_tolerance": bool(abs(delta) <= GATE_TOL_DB),
                "Tolerance_dB": GATE_TOL_DB,
            })
        if repro_rows:
            repro = pd.DataFrame(repro_rows)
            repro_csv = os.path.join(out_dir, "loss_ablation_LARA_mse_reproduction.csv")
            repro.to_csv(repro_csv, index=False)
            print("\n  from-scratch MSE vs published LARA (model_comparison_results.csv)")
            print(repro.to_string(index=False))
            worst = repro["Delta_dB"].abs().max()
            bad = repro[~repro.Within_tolerance]
            if len(bad):
                msg = (f"  worst |delta| = {worst:.4f} dB at CR="
                       f"{int(bad.loc[bad['Delta_dB'].abs().idxmax(), 'CR'])} "
                       f"(tolerance {GATE_TOL_DB} dB)")
                if args.strict:
                    raise SystemExit("ABORT: MSE arm does not reproduce the published run.\n" + msg)
                print("  [warn] " + msg + " -- continue because --strict was not given")
            else:
                print(f"  all CRs within {GATE_TOL_DB} dB; worst |delta| = {worst:.4f} dB")
    elif args.no_gate:
        print("  reproduction check skipped (--no-gate)")
    else:
        print(f"  [warn] published CSV not found at {args.published_csv}; check skipped")

    # ---- figure ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    if not ranking.empty:
        rk = ranking.sort_values("Mean_dSNR_dB")
        axes[0].barh(["MSE (control)"] + rk.Arm.tolist(),
                     [0.0] + rk.Mean_dSNR_dB.tolist(),
                     color=["grey"] + ["steelblue"] * len(rk))
        axes[0].axvline(0.0, color="black", lw=1.0)
        axes[0].set_title("Mean $\\Delta$SNR vs the MSE control (dB)\nunweighted over "
                          f"CR = {CRS}", fontsize=11)
        axes[0].set_xlabel("mean $\\Delta$SNR (dB)")
        axes[0].grid(True, axis="x", alpha=0.3)
        axes[0].set_axisbelow(True)
    else:
        axes[0].text(0.5, 0.5, "no non-MSE arm was run", ha="center", va="center",
                     transform=axes[0].transAxes)

    piv = df.pivot_table(index="Arm", columns="CR", values="SNR")
    order = [a[0] for a in arms if a[0] in piv.index]
    piv = piv.loc[order]
    for cr in piv.columns:
        axes[1].plot(piv.index.tolist(), piv[cr].tolist(), marker="o", label=f"CR={int(cr)}")
    axes[1].set_title("SNR per arm per CR (LARA, from scratch)", fontsize=11)
    axes[1].set_ylabel("SNR (dB)")
    axes[1].tick_params(axis="x", rotation=30)
    axes[1].legend(fontsize=7, ncol=2)
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig_path = os.path.join(out_dir, "loss_ablation_LARA_summary.png")
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)
    print(f"  wrote {fig_path}")

    # ---- verdict ----
    print("\n" + "=" * 78)
    print(f"total {(time.time() - t_all) / 3600.0:.2f} h for {len(df)} runs")
    print("=" * 78)
    mse_snr = df[df.Arm == "MSE"].set_index("CR")["SNR"]
    if ranking.empty:
        print("only the MSE control was run; nothing to rank")
    else:
        best = ranking.iloc[0]
        total_wins = 0
        for name in ranking.Arm:
            total_wins += int((df[df.Arm == name].set_index("CR")["SNR"] > mse_snr).sum())
        print(f"best arm by mean dSNR       : {best.Arm}  ({best.Mean_dSNR_dB:+.3f} dB, "
              f"wins {int(best.CRs_won_vs_MSE)}/{len(CRS)} CRs)")
        print(f"arms beating MSE at >=1 CR  : "
              f"{int((ranking.CRs_won_vs_MSE > 0).sum())} of {len(ranking)}")
        print(f"total arm-wins over MSE     : {total_wins} out of {len(ranking) * len(CRS)}")
        print(f"any arm beats MSE everywhere: "
              f"{bool((ranking.CRs_won_vs_MSE == len(CRS)).all())}")
        if total_wins == 0:
            print("  -> MSE is not beaten at any CR by any arm.")
    print("=" * 78)
    print("Report the UNWEIGHTED numbers and the win counts in the paper.")
    print("The legacy weighted metric is kept in the CSV for continuity only.")


if __name__ == "__main__":
    main()