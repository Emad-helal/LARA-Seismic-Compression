"""
LARA System Block Diagram  –  built from HARA_compact.py (v7)
================================================================================
Renders the LARA (Light Autoencoder for Reconstruction & Analysis) architecture
exactly as trained in Final_LARA_27092026.ipynb, variant = "pyramid_funnel32".

Changes vs. HARA_compact.py
---------------------------
  • Title: HARA -> LARA.  The "(Heavy Capacity)" / "(Light Capacity)" qualifiers
    were dropped from the Encoder and Decoder captions on request; the panel
    titles are now simply "Encoder" and "Decoder".
  • ENCODER: channel widths halved throughout.
        stem       1 ->  64            becomes   1 ->  32
        stages   64->128->256->512     becomes  32-> 64->128->256
    A 1x1 "bottleneck" convolution (bot_conv) is added in front of the pool,
    which HARA did not have.
  • BOTTLENECK PANEL: the single "Compressed Code (C, 1500/CR)" block becomes
    the LARA bottleneck  z = Linear(B*T -> 1500/CR)  with **T = L**
    (no min(latent, L) clamp), so the projection width stays wide as CR grows.
    The latent is a FLAT vector, not a channel tensor.  The panel now shows only
    the latent block and the CR table; the funnel spec lines (B, L = T, widths
    6016 / 3008) were removed on request and live only in this docstring.
  • DECODER: HARA's "Upsample + Residual Block + Attention" chain is replaced by
    LARA's 6-LEVEL CHANNEL PYRAMID (nearest upsample + Conv1d k5 + BN + ELU,
    channels ramp 32 -> 64 and stay at the ceiling of 64), followed by a
    2-layer refinement head: Conv1d 64->32 (k5) + ELU, then Conv1d 32->1 (k15)
    + Sigmoid.  The xN badge becomes "x6 levels".
  • The CR -> latent-points table is UNCHANGED: LARA and HARA both use
    1500 // cr, so the ten widths (750, 500, 300, 150, 100, 75, 50, 30, 25, 15)
    are identical.
  • FIXED: panel B's formula had mojibake ("a" instead of the Greek alpha);
    it is now "alpha".
  • FIXED: the Compress / Expand arrows were misaligned - Compress entered the
    latent panel far below its block, and Expand entered the decoder at its
    bottom row. They now enter at the funnel block's centre and at the
    decoder's first row, so the flow reads left-to-right correctly.
  • FIXED: the base script wrote only a CWD-relative .eps, with no output
    directory, no .png, and a duplicated "Saved." print. Output now goes to
    <review>/updated_outputs/Figures/ in both formats.

Layout engine, colour palette, grid system and detail panels A and B are
carried over unchanged: LARA reuses EnhancedResidualBlock and the SE-style
Channel Attention Block verbatim, so those two panels are already correct.
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# ── Colours ───────────────────────────────────────────────────────────────────
BLUE   = '#1A56DB'
ORANGE = '#D97706'
GREEN  = '#15803D'
RED    = '#B91C1C'
PURPLE = '#5B21B6'
PINK   = '#9D174D'
TEAL   = '#0F766E'
SLATE  = '#1E3A5F'
DARK   = '#111827'
GREY   = '#4B5563'
BG_ENC = '#EFF6FF'; BG_DEC = '#FFF1F2'
BG_LAT = '#F5F0FF'; BG_DA  = '#F0FDF4'; BG_DB  = '#FFFBEB'

# ── Grid ──────────────────────────────────────────────────────────────────────
N     = 6
BH    = 1.10
BW    = 5.60
VGAP  = 0.52
PADV  = 0.62
PADH  = 0.38

PANEL_H = 2*PADV + N*BH + (N-1)*VGAP

LAT_W  = 3.20
IO_W   = 1.30; IO_H = 1.10
H_GAP  = 0.85
IO_GAP = 0.55

ENC_W  = BW + 2*PADH
DEC_W  = ENC_W
TOTAL  = IO_W + IO_GAP + ENC_W + H_GAP + LAT_W + H_GAP + DEC_W + IO_GAP + IO_W

# Detail panels
FOOT   = 0.35
# Detail-panel height.  Panel A's block row is anchored to the panel TOP
# (INPUT_Y = FOOT + DET_H - 1.90 + mh/2) while its formula is anchored to the
# panel BOTTOM (FOOT + 0.32), so DET_H only closes the MIDDLE gap between them
# -- the ~1.02 gap at the top is fixed by the 1.90 constant and cannot shrink
# here.  Hard floor is DET_H > 3.42 (panel A binds; panel B's is 3.32).
DET_H  = 4.20
DET_GAP = 0.55

PBOT   = FOOT + DET_H + DET_GAP
PTOP   = PBOT + PANEL_H

# Title/legend sit just above PTOP
LEG_Y  = PTOP + 0.55   # legend bottom-y
TTL_Y  = LEG_Y + 0.58  # subtitle y
TTL2_Y = TTL_Y + 0.48  # main title y

# Canvas width = content + small margin each side
W_CANVAS = TOTAL + 1.20   # 0.60 margin each side

# Canvas height = just enough above title
H_CANVAS = TTL2_Y + 0.55

LM     = (W_CANVAS - TOTAL) / 2

IO_IN_X  = LM
ENC_X    = LM + IO_W + IO_GAP
LAT_X    = ENC_X + ENC_W + H_GAP
DEC_X    = LAT_X + LAT_W + H_GAP
IO_OUT_X = DEC_X + DEC_W + IO_GAP

CX = W_CANVAS / 2
assert abs((CX-(ENC_X+ENC_W/2)) - ((DEC_X+DEC_W/2)-CX)) < 1e-9

EBX = ENC_X + PADH; DBX = DEC_X + PADH
ECX = EBX + BW/2;   DCX = DBX + BW/2
LCX = LAT_X + LAT_W/2

def ry(i): return PTOP - PADV - (i+1)*BH - i*VGAP
ROWS = [ry(i) for i in range(N)]

IO_CY  = ROWS[0] + BH/2
IO_BOT = IO_CY - IO_H/2
DEC_CY = ROWS[0] + BH/2          # Expand arrow enters the decoder's FIRST row

# ── Canvas ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(W_CANVAS, H_CANVAS))
ax.set_xlim(0, W_CANVAS); ax.set_ylim(0, H_CANVAS)
ax.axis('off')
fig.patch.set_facecolor('white')

# ── Drawing helpers ───────────────────────────────────────────────────────────
def blk(x, yb, w, h, t1, t2='', col=BLUE, dashed=False,
        fs=11.0, ss=9.2, lw=2.0, r=0.16):
    ls = (0,(5,3)) if dashed else 'solid'
    ax.add_patch(FancyBboxPatch((x,yb), w, h,
        boxstyle=f'round,pad=0.04,rounding_size={r}',
        fc=col, ec=col, lw=lw, ls=ls, zorder=3))
    ty = yb + h*(0.63 if t2 else 0.50)
    ax.text(x+w/2, ty, t1, ha='center', va='center',
            fontsize=fs, fontweight='bold', color='white', zorder=4)
    if t2:
        ax.text(x+w/2, yb+h*0.26, t2, ha='center', va='center',
                fontsize=ss, color='white', alpha=0.93, zorder=4)

def panel(x, yb, w, h, bg, ec, lw=2.2):
    ax.add_patch(FancyBboxPatch((x,yb), w, h,
        boxstyle='round,pad=0.10,rounding_size=0.28',
        fc=bg, ec=ec, lw=lw, zorder=1))

def av(x, y1, y2, col, lw=1.9):
    ax.annotate('', xy=(x,y2), xytext=(x,y1),
        arrowprops=dict(arrowstyle='->', color=col, lw=lw, mutation_scale=15), zorder=7)

def ah(x1, x2, y, col, lw=2.2, lbl=''):
    ax.annotate('', xy=(x2,y), xytext=(x1,y),
        arrowprops=dict(arrowstyle='->', color=col, lw=lw, mutation_scale=18), zorder=7)
    if lbl:
        ax.text((x1+x2)/2, y+0.17, lbl, ha='center', va='bottom',
                fontsize=10, fontweight='bold', color=col)

def line(x0,y0,x1,y1, col, lw=1.8, ls='-'):
    ax.plot([x0,x1],[y0,y1], color=col, lw=lw, ls=ls, zorder=5)

def arrowhead(x, y, col, lw=1.8, direction='right'):
    """Small explicit arrowhead."""
    dx = 0.18 if direction=='right' else -0.18
    dy = 0.18 if direction=='up'    else  0.00
    if direction == 'up': dx=0
    ax.annotate('', xy=(x,y), xytext=(x-dx,y-dy),
        arrowprops=dict(arrowstyle='->', color=col, lw=lw, mutation_scale=13), zorder=7)

def shape_tag(x, y, txt, col):
    ax.text(x, y, txt, ha='center', va='center',
            fontsize=8.5, color=col, style='italic', zorder=8,
            bbox=dict(boxstyle='round,pad=0.22', fc='white', ec=col, lw=0.9, alpha=0.96))

# ═══════════════════════════════════════════════════════════════════════════════
# TITLE & LEGEND  (just above main panels)
# ═══════════════════════════════════════════════════════════════════════════════
ax.text(CX, TTL2_Y, 'Light Autoencoder for Reconstruction & Analysis (LARA)',
        ha='center', va='center', fontsize=18, fontweight='bold', color=DARK)
ax.text(CX, TTL_Y, 'System Block Diagram  ·  1-D Seismic Signal Compression  '
                   '·  Filled-Funnel Bottleneck  (T = L)',
        ha='center', va='center', fontsize=11.5, color=GREY)

LEG = [('Conv / BN / ELU',BLUE),('Attention',ORANGE),
       ('Residual Block',GREEN),('Conditional',RED),
       ('Bottleneck',PURPLE),('Upsample',PINK),
       ('Final Conv / Sigmoid',TEAL),('Input / Output',SLATE)]
stp = (W_CANVAS - 2.4) / len(LEG)
for i,(lb,col) in enumerate(LEG):
    lx = 1.2 + i*stp
    ax.add_patch(FancyBboxPatch((lx, LEG_Y), 0.30, 0.24,
        boxstyle='round,pad=0.02', fc=col, ec=col, lw=0, zorder=5))
    ax.text(lx+0.40, LEG_Y+0.12, lb, ha='left', va='center',
            fontsize=8.8, color=GREY)

# ═══════════════════════════════════════════════════════════════════════════════
# INPUT / OUTPUT
# ═══════════════════════════════════════════════════════════════════════════════
for bx, lb in [(IO_IN_X,'Input\nSignal'), (IO_OUT_X,'Output\nSignal')]:
    blk(bx, IO_BOT, IO_W, IO_H, lb, col=SLATE, fs=10.0, lw=2.4, r=0.22)
    shape_tag(bx+IO_W/2, IO_BOT-0.30, '(1, 1500)', SLATE)
ah(IO_IN_X+IO_W, ENC_X-0.06, IO_CY, SLATE, lw=2.1)
ah(DEC_X+DEC_W+0.06, IO_OUT_X, IO_CY, SLATE, lw=2.1)

# ═══════════════════════════════════════════════════════════════════════════════
# ENCODER PANEL
# ═══════════════════════════════════════════════════════════════════════════════
panel(ENC_X, PBOT, ENC_W, PANEL_H, BG_ENC, BLUE)
ax.text(ENC_X+ENC_W/2, PTOP+0.14, 'Encoder',
        ha='center', va='bottom', fontsize=13, fontweight='bold', color=BLUE)

ENC_ROWS = [
    ('Conv1d  +  BN  +  ELU',
     'Channels: 1 → 32,   kernel = 15,   stride = 2',
     BLUE,   '(32, 750)', False),
    ('Channel Attention Block',
     'Input channels:  C = 32',
     ORANGE, '',          False),
    ('Residual Block 1  +  Attention',
     'Channels: 32 → 64,   stride = 2',
     GREEN,  '(64, 375)',False),
    ('Residual Block 2  +  Attention',
     'Channels: 64 → 128,   stride = 2',
     GREEN,  '(128, 188)',False),
    ('Residual Block 3  +  Attention',
     'Channels: 128 → 256,   stride = 2   (CR > 30 only)',
     RED,    '(256, 94)', True),
    ('1×1 Conv  +  ELU  +  AdaptiveAvgPool1d',
     '→  B = 32 channels,   length →  T = L',
     PURPLE, '',          False),
]
for i,(t1,t2,col,stag,dsh) in enumerate(ENC_ROWS):
    blk(EBX, ROWS[i], BW, BH, t1, t2, col, dashed=dsh, fs=10.4, ss=8.9)
    if stag:
        tag_x = ENC_X + ENC_W + 0.22
        tag_y = ROWS[i] + BH/2
        shape_tag(tag_x, tag_y, stag, col)
    if i < N-1: av(ECX, ROWS[i], ROWS[i+1]+BH, col)

# ── Bottleneck panel geometry (needed before the Compress arrow can be placed) ──
LBH = 0.88; LBW = LAT_W - 0.46
LBX = LAT_X + 0.23
# Vertically centre the bottleneck group (latent block + CR table) in the panel.
# The base script anchored the block at the panel centre and hung the table below
# it, which left the top half of the panel empty.  _LAT_GROUP_DROP is tuned for
# the current table offset (ty0 = LBY - 0.62): if that gap changes, this must be
# re-tuned by the same amount, or the group goes off-centre.
_LAT_GROUP_DROP = 1.7625     # group centre sits this far below LBY
LAT_CY = (PTOP + PBOT)/2 + _LAT_GROUP_DROP
LBY    = LAT_CY - LBH/2

# No text label on this arrow: the encoder's shape tags occupy the gap between the
# two panels, and any label here would collide with them.
ah(EBX+BW, LAT_X-0.06, LAT_CY, PURPLE, lw=2.4)

# ═══════════════════════════════════════════════════════════════════════════════
# BOTTLENECK PANEL
# Holds the transmitted latent and the CR -> latent-points table.  The funnel
# arithmetic (B = BOTTLE_CHANNELS = 32, L = T, widths 6016 / 3008) was removed
# from this panel on request; it now lives only in the code and this docstring.
# ═══════════════════════════════════════════════════════════════════════════════
panel(LAT_X, PBOT, LAT_W, PANEL_H, BG_LAT, PURPLE)
ax.text(LCX, PTOP+0.14, 'Bottleneck',
        ha='center', va='bottom', fontsize=13, fontweight='bold', color=PURPLE)

blk(LBX, LBY, LBW, LBH, 'Latent  z', 'Shape:  (1500 / CR,)   flat vector',
    PURPLE, fs=10.0, ss=8.2, lw=2.4, r=0.20)

# CR -> latent points.  Unchanged: LARA and HARA both use 1500 // cr.
crs=[('CR','Pts'),
     ('2','750'),('3','500'),('5','300'),('10','150'),('15','100'),
     ('20','75'),('30','50'),('50','30'),('60','25'),('100','15')]
cw=(LAT_W-0.30)/2; rh=0.255; tx0=LAT_X+0.15; ty0=LBY-0.62
ax.text(LCX,ty0+0.10,'Latent lengths per CR:',ha='center',va='bottom',
        fontsize=8.5,fontweight='bold',color=PURPLE)
for ri,(a,b) in enumerate(crs):
    hdr=(ri==0); bg=PURPLE if hdr else ('#DDD5F5' if ri%2 else '#EDE9FA')
    tc='white' if hdr else DARK
    for ci,val in enumerate([a,b]):
        rx=tx0+ci*cw; ry_=ty0-ri*rh-rh+0.02
        ax.add_patch(FancyBboxPatch((rx,ry_),cw-0.04,rh-0.04,
            boxstyle='round,pad=0.02,rounding_size=0.04',
            fc=bg,ec=PURPLE,lw=0.5 if not hdr else 0,zorder=3))
        ax.text(rx+cw/2-0.02,ry_+rh/2-0.02,val,ha='center',va='center',
                fontsize=8.2,fontweight='bold' if hdr else 'normal',color=tc)

ah(LAT_X+LAT_W+0.06, DBX, DEC_CY, PINK, lw=2.4, lbl='Expand')

# ═══════════════════════════════════════════════════════════════════════════════
# DECODER PANEL  —  6-level channel pyramid + refinement head
# ═══════════════════════════════════════════════════════════════════════════════
panel(DEC_X, PBOT, DEC_W, PANEL_H, BG_DEC, PINK)
ax.text(DEC_X+DEC_W/2, PTOP+0.14, 'Decoder',
        ha='center', va='bottom', fontsize=13, fontweight='bold', color=PINK)
DEC_ROWS = [
    ('Upsample (nearest)  +  Conv1d (k = 5)','Level 1  —  Channels: 32 → 64,   len → 266',PINK),
    ('Upsample (nearest)  +  Conv1d (k = 5)','Level 2  —  Channels: 64,   len → 376',PINK),
    ('Upsample (nearest)  +  Conv1d (k = 5)','Level 3  —  Channels: 64,   len → 531',PINK),
    ('Upsample (nearest)  +  Conv1d (k = 5)','Level N  —  Channels: 64,   len → 1500',PINK),
    ('Conv1d  (refine)  +  ELU','Channels: 64 → 32,   kernel = 5',TEAL),
    ('Conv1d  (Final)  +  Sigmoid','Channels: 32 → 1,   kernel = 15',TEAL),
]
for i,(t1,t2,col) in enumerate(DEC_ROWS):
    blk(DBX,ROWS[i],BW,BH,t1,t2,col,fs=10.0,ss=8.9)
    if i < N-1: av(DCX,ROWS[i],ROWS[i+1]+BH,col)

# x6 badge spans all 4 shown upsampling levels (rows 0-3)
loop_top=ROWS[0]+BH; loop_bot=ROWS[3]; loop_mid=(loop_top+loop_bot)/2
bx2=DBX+BW+0.12
ax.add_patch(FancyBboxPatch((bx2,loop_mid-0.50),0.74,1.00,
    boxstyle='round,pad=0.05,rounding_size=0.14',fc=PINK,ec=PINK,lw=0,zorder=5))
ax.text(bx2+0.37,loop_mid+0.20,'×6',ha='center',va='center',
    fontsize=18,fontweight='bold',color='white',zorder=6)
ax.text(bx2+0.37,loop_mid-0.18,'levels',ha='center',va='center',
    fontsize=8.5,color='white',fontweight='bold',zorder=6)
for yy in [loop_top+0.04,loop_bot-0.04]:
    ax.plot([DBX+BW+0.06,bx2],[yy,yy],color=PINK,lw=1.4,ls='--',zorder=4)

# ═══════════════════════════════════════════════════════════════════════════════
# DETAIL PANELS
# ═══════════════════════════════════════════════════════════════════════════════
GAP_AB = 0.80
pW  = (W_CANVAS - 2*0.35 - GAP_AB) / 2
pA_x = 0.35;  pB_x = pA_x + pW + GAP_AB
pH   = DET_H

panel(pA_x, FOOT, pW, pH, BG_DA, GREEN, lw=1.8)
ax.text(pA_x+pW/2, FOOT+pH+0.10, 'A.  Enhanced Residual Block',
        ha='center', va='bottom', fontsize=12.5, fontweight='bold', color=GREEN)

panel(pB_x, FOOT, pW, pH, BG_DB, ORANGE, lw=1.8)
ax.text(pB_x+pW/2, FOOT+pH+0.10, 'B.  Channel Attention Block  (SE-style)',
        ha='center', va='bottom', fontsize=12.5, fontweight='bold', color=ORANGE)

# ═══════════════════════════════════════════════════════════════════════════════
# PANEL A  –  Enhanced Residual Block
# Layout (left → right):
#   "x" input  →  [Conv1d k3] → [BN+ELU] → [Conv1d k3] → [BN]  →  ⊕  →  [ELU]  →  [Attn]
#              ↓ (branch down)
#           [Conv1d k=1 + BN  or  Identity]
#              └──────────────────────────────────────────────────────────────↑ (up into ⊕)
# ═══════════════════════════════════════════════════════════════════════════════
mw = 1.32; mh = 0.88; mg = 0.22          # mini block width/height/gap
mblks = [('Conv1d\nk = 3', BLUE),
         ('BN + ELU',       BLUE),
         ('Conv1d\nk = 3', BLUE),
         ('BN',             BLUE)]
tw   = len(mblks)*mw + (len(mblks)-1)*mg  # total main-path width

# Anchor: x input point
INPUT_X = pA_x + 0.55
INPUT_Y = FOOT + pH - 1.90 + mh/2        # vertical centre of main-path row

# Main-path block row
mx0 = INPUT_X + 0.55   # first block left edge
my  = INPUT_Y - mh/2   # block bottom

# Sum node
sx  = mx0 + tw + 0.34   # sum node x centre

# Post-sum blocks
ELU_X   = sx + 0.50
ELU_W   = 0.80
ATTN_X  = ELU_X + ELU_W + 0.22
ATTN_W  = 1.55

# Shortcut block
SC_H  = 0.68
SC_Y  = my - 1.05            # shortcut block bottom y
SC_X  = mx0                  # aligned under first main-path block
SC_W  = sx - mx0 - 0.10     # spans from first block to just under ⊕

# --- Draw main-path label
ax.text(pA_x+0.28, INPUT_Y, 'x',
        ha='center', va='center', fontsize=18, fontweight='bold', color=DARK)
# Arrow from x into first block
ah(INPUT_X, mx0, INPUT_Y, BLUE, lw=1.8)

# Draw main-path blocks + inter-block arrows
for i,(lb,col) in enumerate(mblks):
    bx = mx0 + i*(mw+mg)
    blk(bx, my, mw, mh, lb, col=col, fs=10.0, lw=1.6, r=0.12)
    if i < len(mblks)-1:
        ah(bx+mw, bx+mw+mg, INPUT_Y, col, lw=1.6)

# Arrow: last main-path block → ⊕
ah(mx0+tw, sx-0.27, INPUT_Y, BLUE, lw=1.6)

# Sum circle
ax.add_patch(plt.Circle((sx, INPUT_Y), 0.27, color=GREEN, zorder=5))
ax.text(sx, INPUT_Y, '⊕', ha='center', va='center',
        fontsize=17, color='white', fontweight='bold', zorder=6)

# Arrow: ⊕ → ELU
ah(sx+0.27, ELU_X, INPUT_Y, GREEN, lw=1.6)
blk(ELU_X, my, ELU_W, mh, 'ELU', col=BLUE, fs=10.0, lw=1.6, r=0.12)

# Arrow: ELU → Attention
ah(ELU_X+ELU_W, ATTN_X, INPUT_Y, ORANGE, lw=1.6)
blk(ATTN_X, my, ATTN_W, mh, 'Attention\nBlock', col=ORANGE, fs=10.0, lw=1.6, r=0.12)

# --- Shortcut path labels
ax.text(pA_x+0.28, SC_Y+SC_H/2, 'Short-\ncut', ha='center', va='center',
        fontsize=9.5, fontweight='bold', color=GREY)

# Shortcut block
blk(SC_X, SC_Y, SC_W, SC_H,
    'Conv1d (k=1) + BN   or   Identity',
    col='#546E7A', fs=10.0, lw=1.6, r=0.12)

# Branch DOWN from input junction (just left of first block)
BRANCH_X = INPUT_X + 0.28
line(BRANCH_X, INPUT_Y, BRANCH_X, SC_Y+SC_H/2, GREY, lw=1.8, ls='--')
# Arrow RIGHT into shortcut block
ah(BRANCH_X, SC_X, SC_Y+SC_H/2, GREY, lw=1.8)

# Arrow RIGHT from shortcut block to below ⊕, then UP
line(SC_X+SC_W, SC_Y+SC_H/2, sx, SC_Y+SC_H/2, GREY, lw=1.8, ls='--')
# Arrow UP from shortcut level to ⊕ bottom
av(sx, SC_Y+SC_H/2, INPUT_Y-0.27, GREY, lw=1.8)

# Formula
ax.text(pA_x+pW/2, FOOT+0.32,
        'y = ELU( F(x) + shortcut(x) )     →     ŷ = Attention(y)',
        ha='center', va='center', fontsize=11.0, fontweight='bold', color=GREEN)

# ═══════════════════════════════════════════════════════════════════════════════
# PANEL B  –  Channel Attention Block (SE-style)
# Layout: [Input x] → [AvgPool] → [Conv C→C/8] → [ReLU] → [Conv C/8→C] → [Sigmoid] → [⊗]
#          |                                                                              ↑
#          └──────────────────── bypass (x) ────────────────────────────────────────────┘
# ═══════════════════════════════════════════════════════════════════════════════
ab_blks = [('Input  x',        SLATE,  1.10),
           ('AvgPool1d\n(Global)', PURPLE, 1.30),
           ('Conv1d\nC→max(4,C//8)', BLUE, 1.65),
           ('ReLU',            BLUE,   0.90),
           ('Conv1d\nmax(4,C//8)→C', BLUE, 1.65),
           ('Sigmoid  σ',      BLUE,   1.20),
           ('⊗\n(x · w)',      ORANGE, 1.00)]
ag = 0.20
at = sum(w for _,_,w in ab_blks) + (len(ab_blks)-1)*ag
ah2 = 0.92
ab_x0 = pB_x + (pW - at) / 2
ab_y  = FOOT + pH - 1.90
ab_cy = ab_y + ah2/2

# Draw SE-path blocks + arrows
for i,(lb,col,aw_i) in enumerate(ab_blks):
    bx = ab_x0 + sum(w for _,_,w in ab_blks[:i]) + i*ag
    blk(bx, ab_y, aw_i, ah2, lb, col=col, fs=10.0, lw=1.6, r=0.14)
    if i < len(ab_blks)-1:
        next_bx = bx + aw_i
        ah(next_bx, next_bx+ag, ab_cy, ORANGE, lw=1.7)

# Bypass line routing:
#   ┌── down from Input x bottom-centre
#   └── right along BP_Y  (plain line, no arrowhead)
#                            └── UP with arrowhead into ⊗ bottom
BP_START_X = ab_x0 + ab_blks[0][2]/2                          # centre of "Input x"
BP_END_X   = ab_x0 + sum(w for _,_,w in ab_blks[:-1]) + (len(ab_blks)-1)*ag + ab_blks[-1][2]/2  # centre of "⊗"
BP_Y       = ab_y - 0.62                               # horizontal rail below blocks

# 1. Vertical DOWN from Input x block bottom → rail
line(BP_START_X, ab_y, BP_START_X, BP_Y, SLATE, lw=2.0)

# 2. Horizontal RIGHT along rail (plain line, dot at corner)
line(BP_START_X, BP_Y, BP_END_X, BP_Y, SLATE, lw=2.0)

# 3. Vertical UP with arrowhead from rail into ⊗ block bottom
av(BP_END_X, BP_Y, ab_y, SLATE, lw=2.0)

# Corner dots to mark junctions cleanly
ax.plot(BP_START_X, BP_Y, 'o', color=SLATE, ms=5, zorder=8)
ax.plot(BP_END_X,   BP_Y, 'o', color=SLATE, ms=5, zorder=8)

ax.text((BP_START_X+BP_END_X)/2, BP_Y-0.13,
        'Input bypass  (x)',
        ha='center', va='top', fontsize=10.0, color=SLATE, style='italic')

# Formula   (alpha fixed: the base script had mojibake "â" here)
ax.text(pB_x+pW/2, FOOT+0.32,
        'α = σ( W₂ · ReLU( W₁ · AvgPool(x) ) )           x̂ = x · α     where  W₁: C→max(4,C//8),  W₂: max(4,C//8)→C',
        ha='center', va='center', fontsize=10.0, fontweight='bold', color=ORANGE)

# ═══════════════════════════════════════════════════════════════════════════════
# SAVE   →   <review>/updated_outputs/Figures/
# ═══════════════════════════════════════════════════════════════════════════════
HERE    = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(os.path.dirname(HERE), 'updated_outputs', 'Figures')
os.makedirs(OUT_DIR, exist_ok=True)

out_png = os.path.join(OUT_DIR, 'Final_LARA_pipeline.png')
out_eps = os.path.join(OUT_DIR, 'Final_LARA_pipeline.eps')
plt.savefig(out_png, format='png', dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig(out_eps, format='eps', bbox_inches='tight', facecolor='white')
print("Saved:", out_png)
print("Saved:", out_eps)
