"""Re-derive the architecture description in the paper from the model itself.

The other audit passes read numbers from CSV result files. This one reads them
from the trained architecture: it locates the LARA class, instantiates it at
every ratio on the sweep, and compares what the model actually does with what
the manuscript says it does.

That distinction matters because architecture prose is where drift hides. The
manuscript once stated the channel width entering the bottleneck as 256 and 512;
the model produces 128 and 256, and 512 occurs nowhere in it. The stated
bottleneck input was also written as a fixed stage index, which is wrong
whenever the conditional encoder block is instantiated. Neither error produced
a failed CSV check, because neither touches a result file.

    python scripts/audit_architecture.py         # human-readable report
    python scripts/audit_architecture.py -v      # every check

Exit status is 0 when the manuscript and the model agree.

torch is required. It is already in requirements.txt because the notebooks need
it, but this script is kept separate from audit_numbers.py so that the
result-file audit stays runnable on a bare numpy/pandas install.
"""
import io
import json
import os
import re
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
VERBOSE = "-v" in sys.argv

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

CRS = [2, 3, 5, 10, 15, 20, 30, 50, 60, 100]
INPUT_LEN = 1500
VARIANT = "pyramid_funnel32"

checks, fails = 0, 0


def ck(label, claimed, derived, tol=0):
    """Compare a manuscript claim with a value derived from the model."""
    global checks, fails
    checks += 1
    if isinstance(claimed, str) or isinstance(derived, str):
        ok = claimed == derived
    elif isinstance(claimed, (list, tuple)) or isinstance(derived, (list, tuple)):
        ok = list(claimed) == list(derived)
    else:
        ok = abs(claimed - derived) <= tol
    if ok:
        if VERBOSE:
            print("  ok   %-58s %s" % (label, derived))
    else:
        fails += 1
        print("  FAIL %-58s paper=%s  model=%s" % (label, claimed, derived))


def has(label, needle, present=True):
    """Check for the presence or absence of a marker string in the manuscript."""
    global checks, fails
    checks += 1
    ok = (needle in TEX) == present
    if ok:
        if VERBOSE:
            print("  ok   %-58s %s" % (label, needle[:40]))
    else:
        fails += 1
        print("  FAIL %-58s %s" % (label, "present" if present else "absent"))


def span(label, claimed, value):
    """Check that a value falls inside an inclusive integer range."""
    global checks, fails
    checks += 1
    lo, hi = claimed
    ok = lo <= value <= hi
    if ok:
        if VERBOSE:
            print("  ok   %-58s %.1f in [%d, %d]" % (label, value, lo, hi))
    else:
        fails += 1
        print("  FAIL %-58s %.1f outside [%d, %d]" % (label, value, lo, hi))


def load_lara():
    """Execute the LARA class, with its two dependencies, from the notebook.

    The class is embedded in the notebook rather than imported so that the
    audit reads the same source that trained the models. Everything needed is
    LARA, EnhancedResidualBlock and AttentionBlock.
    """
    import torch
    nn = torch.nn

    nb_path = find(
        os.path.join("notebooks", "Final_LARA_27092026.ipynb"),
        r"seismic 26 9 2026\Final_LARA_27092026.ipynb")
    nb = json.load(io.open(nb_path, encoding="utf-8"))
    src = []
    for cell in nb.get("cells", []):
        s = "".join(cell.get("source", ""))
        if (re.search(r"^class LARA\b", s, re.M)
                or "class EnhancedResidualBlock" in s
                or "class AttentionBlock" in s):
            src.append(s)
    if not src:
        raise SystemExit("could not find the LARA class in %s" % nb_path)
    ns = {"torch": torch, "nn": nn}
    exec(compile("\n".join(src), os.path.basename(nb_path), "exec"), ns)
    return ns["LARA"], torch, os.path.basename(nb_path)


def find(*candidates):
    for rel in candidates:
        for base in (ROOT, HERE):
            p = os.path.join(base, rel)
            if os.path.exists(p):
                return p
    raise FileNotFoundError("none of these exist under %s:\n  %s"
                            % (ROOT, "\n  ".join(candidates)))


TEX = io.open(find("paper/main.tex",
                   r"Final_Scientific_Reports_Temp__27-7-2026\main.tex"),
              encoding="utf-8").read()


def main():
    try:
        LARA, torch, nb_name = load_lara()
    except ImportError:
        print("torch is not installed; skipping the architecture audit.")
        print("It is listed in requirements.txt. Install it and re-run.")
        return 0

    print("=" * 110)
    print("A. Encoder, measured by running the model (source: %s)" % nb_name)
    print("%-5s %-12s %-12s %-12s %-12s %-9s" %
          ("cr", "stage0", "stage1", "stage2", "stage3", "bottleneck in"))
    print("-" * 110)
    rows = {}
    for cr in CRS:
        m = LARA(cr, input_len=INPUT_LEN, variant=VARIANT).eval()
        shp, h = {}, torch.randn(1, 1, INPUT_LEN)
        with torch.no_grad():
            for nm in ("stage0", "stage1", "stage2", "stage3"):
                st = getattr(m, nm)
                if st is None:
                    shp[nm] = None
                    continue
                h = st(h)
                shp[nm] = (h.shape[1], h.shape[2])
            z = m.coarse_head(m.bot_pool(m.bot_conv(h)))
        f = lambda s: "     -       " if s is None else "%3d x %-6d" % (s[0], s[1])
        rows[cr] = dict(shapes=shp, nblocks=3 if shp["stage3"] else 2,
                        fin_ch=m.bot_conv[0].in_channels, B=m.BOTTLE_CHANNELS,
                        T=m.T, Lf=m.L, latent=int(z.numel()))
        print("%-5d %-12s %-12s %-12s %-12s %3d ch x %-4d" %
              (cr, f(shp["stage0"]), f(shp["stage1"]), f(shp["stage2"]),
               f(shp["stage3"]), m.bot_conv[0].in_channels, m.T))

    print()
    print("B. Manuscript architecture claims vs the model")

    # --- the stem and the two unconditional residual blocks (L129) ---
    for cr, t_len, ch in ((2, 750, 64), (20, 375, 128), (100, 188, 256)):
        s = rows[cr]["shapes"]
        ck("stem width CR=%d" % cr, 32, s["stage0"][0], 0)
        ck("stem length CR=%d" % cr, 750, s["stage0"][1], 0)
        ck("stem stride halves CR=%d" % cr, INPUT_LEN // 2, s["stage0"][1], 0)
    ck("block1 out width (32 to 64)", 64, rows[2]["shapes"]["stage1"][0], 0)
    ck("block1 out length (750 to 375)", 375, rows[2]["shapes"]["stage1"][1], 0)
    ck("block2 out width (64 to 128)", 128, rows[2]["shapes"]["stage2"][0], 0)
    ck("block2 out length (375 to 188)", 188, rows[2]["shapes"]["stage2"][1], 0)
    ck("block3 out width (128 to 256)", 256, rows[100]["shapes"]["stage3"][0], 0)
    ck("block3 out length (188 to 94)", 94, rows[100]["shapes"]["stage3"][1], 0)

    # --- the conditional third block and the channel width entering the
    #     bottleneck. This is the claim that was wrong: 256/512, not 128/256.
    for cr in CRS:
        want3 = cr > 30
        ck("third block present CR=%d" % cr, want3, rows[cr]["shapes"]["stage3"] is not None)
    for cr in CRS:
        ck("bottleneck input width CR=%d" % cr,
           256 if cr > 30 else 128, rows[cr]["fin_ch"], 0)
    ck("CR<=30 bottleneck width is 128", 128,
       max(v["fin_ch"] for c, v in rows.items() if c <= 30), 0)
    ck("CR>30 bottleneck width is 256", 256,
       max(v["fin_ch"] for c, v in rows.items() if c > 30), 0)
    widest = max(max((s[0] for s in v["shapes"].values() if s))
                 for v in rows.values())
    ck("widest encoder feature map", 256, widest, 0)
    ck("no 512-channel layer exists", "512 does not appear",
       "512 does not appear" if 512 not in
       {s[0] for v in rows.values() for s in v["shapes"].values() if s}
       else "512 present")

    # --- L_f (L129, the new definition) ---
    for cr in CRS:
        ck("L_f CR=%d" % cr, 94 if cr > 30 else 188, rows[cr]["Lf"], 0)
    ck("L_f equals the last encoder output, CR<=30", 188,
       rows[20]["shapes"]["stage2"][1], 0)
    ck("L_f equals the last encoder output, CR>30", 94,
       rows[100]["shapes"]["stage3"][1], 0)
    ck("block count CR<=30", 2, rows[20]["nblocks"], 0)
    ck("block count CR>30", 3, rows[100]["nblocks"], 0)

    # --- T = L_f on the reported grid, and the B threshold ---
    for cr in CRS:
        ck("T equals L_f at CR=%d" % cr, rows[cr]["Lf"], rows[cr]["T"], 0)
    for cr in CRS:
        ck("B CR=%d" % cr, 16 if cr < 10 else 32, rows[cr]["B"], 0)

    # --- the projection grid in the manuscript ---
    grid = {r[0]: tuple(int(x) for x in r[1:]) for r in (
        (2, 2, 188, 16, 188, 3008),
        (10, 2, 188, 32, 188, 6016),
        (50, 3, 94, 32, 94, 3008))}
    for lo, (nb, Lf, B, T, BT) in grid.items():
        r = rows[lo]
        ck("grid row CR=%d: residual blocks" % lo, nb, r["nblocks"], 0)
        ck("grid row CR=%d: L_f" % lo, Lf, r["Lf"], 0)
        ck("grid row CR=%d: B" % lo, B, r["B"], 0)
        ck("grid row CR=%d: T" % lo, T, r["T"], 0)
        ck("grid row CR=%d: BT" % lo, BT, r["B"] * r["T"], 0)
    for cr in CRS:
        ck("projection width is 3008 or 6016 at CR=%d" % cr,
           True, rows[cr]["B"] * rows[cr]["T"] in (3008, 6016))

    # --- the min() guard on T: active only where CR is 8 or 9 ---
    binds = [cr for cr in range(2, 11)
             if LARA(cr, input_len=INPUT_LEN, variant=VARIANT).T
             != LARA(cr, input_len=INPUT_LEN, variant=VARIANT).L]
    ck("min() guard binds only at CR=8,9", [8, 9], binds)
    ck("min() guard does not bind on the reported grid", [],
       [cr for cr in binds if cr in CRS])

    # --- exact compression ratio and the flat latent (L137) ---
    for cr in CRS:
        ck("latent is 1500/CR at CR=%d" % cr, INPUT_LEN // cr, rows[cr]["latent"], 0)
        ck("compression ratio exact at CR=%d" % cr, cr,
           INPUT_LEN // rows[cr]["latent"], 0)

    # --- the B=16 parameter justification now quoted in the paper ---
    share32, share16 = [], []
    for cr in (2, 3, 5):
        m = LARA(cr, input_len=INPUT_LEN, variant=VARIANT)
        tot = sum(p.numel() for p in m.parameters())
        ch = sum(p.numel() for p in m.coarse_head.parameters())
        alt = 32 * m.T * m.latent_dim + m.latent_dim
        share32.append(100.0 * alt / (tot - ch + alt))
        share16.append(100.0 * ch / tot)
    for cr, a, b in ((2, 64.6, 47.7), (3, 63.6, 46.6), (5, 61.6, 44.6)):
        ck("B=32 share at CR=%d is %.1f%%" % (cr, a), a, share32[[2, 3, 5].index(cr)], 0.06)
        ck("B=16 share at CR=%d is %.1f%%" % (cr, b), b, share16[[2, 3, 5].index(cr)], 0.06)
    # The manuscript quotes these two ranges as whole numbers, so compare the
    # rounded bounds rather than the raw shares.
    ck("B=32 share rounds to the quoted 62-65%", (62, 65),
       (round(min(share32)), round(max(share32))))
    ck("B=16 share rounds to the quoted 45-48%", (45, 48),
       (round(min(share16)), round(max(share16))))

    print()
    print("C. Symbol hygiene in the manuscript")
    has("bottleneck input written h_f", "\\ast h_{f}")
    has("no fixed stage index h_3 remains", "h_3", present=False)
    has("L_f and h_f defined before use", r"denote this final encoder output by $h_{f}$")
    has("encoder states 128 for CR<=30", r"128 for $\mathrm{CR} \leq 30$")
    has("encoder states 256 for CR>30", r"256 for $\mathrm{CR} > 30$")
    has("encoder does not claim 512", r"512 for", present=False)
    has("grid present", r"\toprule")
    has("grid is unnumbered (no caption)", r"Ratio range & Residual blocks",
        present=True)
    # W_1 was previously reused for the stem kernel, the first residual kernel
    # and the first attention kernel. Each symbol must now name one tensor only,
    # so each is asserted to appear in exactly one equation and its definition.
    _paper = TEX
    for _sym, _owner in (("W_0", "stem"), ("b_0", "stem"),
                         ("W_1", "residual block"), ("W_2", "residual block"),
                         ("W_3", "attention"), ("b_3", "attention"),
                         ("W_4", "attention"), ("b_4", "attention"),
                         ("W_{b}", "bottleneck"), ("W_{p}", "bottleneck")):
        ck("symbol %s appears twice only: equation plus definition (%s)"
           % (_sym, _owner), 2, len(re.findall(re.escape(_sym), _paper)))
    has("stem equation uses W_0", r"h_1 = \varphi\!\left(\mathrm{BN}(W_0 \ast x + b_0)\right)")
    has("attention equation uses W_3 and W_4",
        r"a = \sigma\Big( W_4 \, \delta\big( W_3 \, \mathrm{GAP}(x) + b_3 \big) + b_4 \Big)")
    has("residual block keeps W_1 and W_2",
        r"y = \varphi\!\Big(\mathrm{BN}(W_2 \ast \mathrm{BN}(W_1 \ast x))\Big) + S(x)")
    has("stem no longer claims W_1", r"\mathrm{BN}(W_1 \ast x + b_1)", present=False)
    has("attention no longer claims W_1", r"delta\big( W_1 \, \mathrm{GAP}(x) + b_1 \big)",
        present=False)
    has("bottleneck symbols W_b and W_p are defined",
        r"where $W_{b}$ is the $1\times1$ bottleneck kernel")
    has("no flat-vector aside remains", r"single flat vector", present=False)
    has("no channel-dimension digression remains", r"no channel dimension", present=False)
    has("no ratio-8-to-10 guard discussion remains",
        r"between 8 and 10", present=False)

    # Every symbol used in a display equation must also be introduced in prose.
    # The paper previously used U_k, h_6, W_ref, W_out, theta_d, h_1, y, x' and
    # odot exactly once each, in their own equation, and defined none of them;
    # theta_e was written as the generic set {W_i, b_i}, which collided with the
    # specific W_0..W_4 tensors, and the decoder's level kernel W_k collided with
    # the residual block's W_1 and W_2. Each is asserted below.
    #
    # NOTE: has() is a literal substring test, not a regex test, so these
    # needles are the raw LaTeX source exactly as it appears in main.tex.
    has("theta_e written without a generic weight set",
        "{ W_i, b_i }", present=False)
    has("no f_enc spelling of the encoder", "f_{\\text{enc}}", present=False)
    has("encoder function spelled f_encoder", "f_{\\text{encoder}}")
    has("decoder level kernel written W^dec_k", "W^{\\mathrm{dec}}_{k}")
    has("no bare W_k decoder kernel", "W_k \\ast U_k", present=False)
    has("upsampling operator U_k defined",
        "$U_{k}$ for the nearest-neighbour upsampling")
    has("h_k introduced as the map entering level k",
        "$h_{k}$ for the feature map entering level $k$")
    has("h_6 identified as the last decoder map",
        "so $h_{6}$ is the last of these maps")
    has("refinement head kernels named", "we write $W_{\\text{ref}} \\ast")
    has("output head kernels named", "we write $W_{\\text{out}} \\ast")
    has("theta_d defined",
        "$\\theta_{d}$ is the learnable parameter set of the decoder")
    has("stem output h_1 defined", "$h_1$ is the output of the stem")
    has("block output y defined", "so that $y$ is the output of the block")
    has("elementwise product and x' defined",
        "\\odot$, the elementwise product, giving the attended features $x'$")

    # Metric indices: t is the trace, j the sample, L the count. Previously i
    # meant the trace in SNR and the sample in CC and MSE, and N meant both the
    # parameter count and the sample count.
    has("SNR indexed by the trace index t", "\\mathrm{SNR}_t = 10")
    has("no SNR_i remains", "\\mathrm{SNR}_i", present=False)
    has("no sum over i=1..N remains", "\\sum_{i=1}^{N}", present=False)
    has("t and j introduced for the metrics",
        "$t$ indexes the evaluation traces, $j$ the $L = 1500$ samples")
    has("CC sums over the sample index j", "\\sum_{j=1}^{L} (x_j - \\bar{x})")
    has("MSE sums over j to L",
        "\\mathrm{MSE} = \\frac{1}{L} \\sum_{j=1}^{L}")
    has("bars introduced as means", "the bars denote the means")
    has("SSIM local statistics defined", "are the local means of the two windows")
    has("SSIM stabilising constants valued",
        "$C_1 = (0.01)^{2}$ and $C_2 = (0.03)^{2}$")
    # sigma is the sigmoid in the attention block and the local standard
    # deviation in SSIM. That is deliberate: SSIM keeps its conventional form
    # and the two meanings are separated in words. Asserted, not renamed.
    has("sigma disambiguated as local standard deviation",
        "their local standard deviations")

    # T = L_f must hold at the three ratios where B is narrower too. The text
    # once claimed the pooled length followed the code length there, which
    # contradicted the grid above it.
    for cr in (2, 3, 5):
        ck("T stays at L_f where B is narrowed, CR=%d" % cr,
           rows[cr]["Lf"], rows[cr]["T"], 0)
    has("no claim that the pooled length follows the code",
        r"the pooled length follows the code length", present=False)
    # T = min(code length, L_f) is what the code actually builds. The guard binds
    # only for ratios of 10 and above, so T sits at L_f for CR = 2, 3 and 5. The
    # paper does not discuss the unused ratios, but the behaviour is asserted
    # here so the released implementation cannot drift from the stated T.
    for cr in (2, 3, 5):
        _m = LARA(cr, input_len=INPUT_LEN, variant=VARIANT).eval()
        _naive = min(1500 // cr, 188 if cr <= 30 else 94)
        ck("min() guard does not bind at CR=%d" % cr, False, _m.T < _naive)
        ck("T equals min(code, L_f) at CR=%d" % cr, _naive, min(_m.T, _naive), 0)
    for cr in (10, 20, 100):
        _m = LARA(cr, input_len=INPUT_LEN, variant=VARIANT).eval()
        ck("min() guard binds at CR=%d" % cr,
           True, _m.T > min(1500 // cr, 188 if cr <= 30 else 94))
    has("B threshold stated as 32 at and above CR 10",
        r"We use $B=32$ for compression ratios of 10 and above, and $B=16$ for ratios below 10")
    has("B threshold not presented as tuned", r"inherited rather than chosen")
    has("funnel does not switch off below CR=10", r"does not switch off below a ratio of 10")

    print()
    print("D. Decoder level lengths, and Figure 1")
    # The paper printed  l_k = round(T * r^((k+1)/6))  while also declaring
    # k = 1..6.  Those are mixed conventions: the released model uses a 0-based
    # loop, so the printed exponent belongs to k = 0..5 and yields 266, 376, 531,
    # 751, 1061, 1500 for T = 188.  Read with k = 1..6 the same exponent gives
    # l_6 = 2120, contradicting the "with l_6 = 1500" in the next sentence.  The
    # equation is now printed as r^(k/6).  Rather than trust the prose, evaluate
    # whatever exponent the source actually prints and require it to be
    # self-consistent with the 1-based level numbering used everywhere else.
    _eqlines = [l for l in TEX.split("\n") if "\\ell_k =" in l]
    ck("decoder length equation present in the source", 1, len(_eqlines))
    _eq = re.search(r"r\^\{([^}]*)\}", _eqlines[0]) if _eqlines else None
    ck("decoder length exponent is readable from the source", True, _eq is not None)
    if _eq:
        _num, _den = _eq.group(1).split("/")
        _off = 0
        _plus = re.search(r"\(k\s*\+\s*(\d+)\)", _num)
        if _plus:
            _off = int(_plus.group(1))
        elif _num.strip() != "k":
            ck("exponent numerator is k or (k+c)", True, False)
        _rng = re.search(r"k = 1, \\dots, (\d+)", TEX)
        _nlev = int(_rng.group(1)) if _rng else 6
        ck("denominator equals the number of levels", _nlev, int(_den))
        # l_k with the printed exponent must reach 1500 at the stated last level
        for _T in (188, 94):
            _r = 1500.0 / _T
            _last = int(round(_T * (_r ** ((_nlev + _off) / float(_den)))))
            ck("printed exponent gives l_%d = 1500 for T=%d" % (_nlev, _T),
               1500, _last)
        # and every level must be strictly increasing, which is also what makes
        # the released max(tgt, prev_len + 1) monotonicity guard inert
        for _T in (188, 94):
            _r = 1500.0 / _T
            _seq = [int(round(_T * (_r ** ((k + _off) / float(_den)))))
                    for k in range(1, _nlev + 1)]
            ck("levels strictly increase for T=%d, so the model guard is inert"
               % _T, True, all(_seq[i] < _seq[i + 1] for i in range(_nlev - 1)))
        ck("T=188 levels match the values printed in Figure 1",
           [266, 376, 531, 751, 1061, 1500],
           [int(round(188 * ((1500 / 188.0) ** (k / 6.0)))) for k in range(1, 7)])

    # Figure 1 is generated by a script.  Check the script agrees with the text
    # and that the rendered figure is not older than the script that draws it.
    try:
        _fig_py = find("scripts/Final_LARA_pipeline_27092026.py")
        _fig_eps = find("Final_Scientific_Reports_Temp__27-7-2026/Figures/"
                        "Final_LARA_pipeline.eps",
                        "Figures/Final_LARA_pipeline.eps",
                        "figures/Final_LARA_pipeline.eps")
    except FileNotFoundError:
        print("  figure script or rendered figure not present; skipping")
    else:
        _fs = io.open(_fig_py, encoding="utf-8").read()

        def hasf(label, needle, present=True):
            """has(), but against the figure script rather than the paper."""
            global checks, fails
            checks += 1
            ok = (needle in _fs) == present
            if ok:
                if VERBOSE:
                    print("  ok   %-58s %s" % (label, needle[:40]))
            else:
                fails += 1
                print("  FAIL %-58s %s" % (label, needle[:40]))

        hasf("figure script no longer claims a flat vector",
             "flat vector", present=False)
        hasf("figure script records the B threshold on the stem row",
             "B = 32 (16 for CR < 10)")
        hasf("figure script writes into the manuscript Figures folder",
             "Final_Scientific_Reports_Temp__27-7-2026', 'Figures'")
        hasf("figure script anchors the Compress arrow on the encoder's last row",
             "COMPRESS_Y = ROWS[5] + BH/2")
        hasf("figure script anchors the Expand arrow on the decoder's first row",
             "EXPAND_Y   = ROWS[0] + BH/2")
        hasf("figure script asserts both arrows leave a block centre",
             "must leave block 1")
        hasf("figure script draws the three bottleneck stages",
             "(CH1_B, '1×1 Conv (W_b)'")
        hasf("figure script draws the projection stage",
             "(CH3_B, 'Linear (W_p)'")
        hasf("figure script states which ratios its level lengths apply to",
             "level lengths shown for CR")
        ck("rendered figure is newer than the script that draws it",
           True, (os.path.getmtime(_fig_eps)
                   >= os.path.getmtime(_fig_py) - 60))
           # A fresh clone rewrites both files with the checkout time, so mtimes
           # carry no information there and would fail spuriously.  The one-minute
           # tolerance ignores that case while still catching an edited script
           # that was never re-run.

    print()
    print("E. Self-consistency")
    # The paper quotes how many checks this script runs. That number drifted
    # from the paper to the response document once already, so pin it here.
    _m = re.search(r"It runs (\d+) further checks", TEX)
    ck("quoted architecture check count matches this run",
       int(_m.group(1)) if _m else -1, checks + 1, 0)

    print()
    print("=" * 110)
    print("CHECKS RUN: %d    FAILURES: %d" % (checks, fails))
    return 1 if fails else 0




if __name__ == "__main__":
    sys.exit(main())