"""The thumbnail of the README (assets/thumbnail.png).

Two observers explored the same object (trial 123: bumpy, physical metallicity
0.5, Sky environment, head and hand) and answered differently: observer 16
looked at it against the sea and said "metal", observer 1 held it up into the
sun and said "glass".  The top rows show frames of the two first-person videos; the
right-hand column shows what the model does with them: 116 statistics of each
video, a logistic regression, and the predicted metallicity of each video on
the model's curve.  The statistics and predicted metallicities are the real
ones (results/features/, results/models/); the frames are read from data/imgs,
which is not distributed, so this script is kept for the record.

Run from anywhere:  python assets/make_thumbnail.py
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from common import IMGS, RESULTS, MAIN_FEATURES, context_crop, frame_paths     # noqa: E402
from evaluate import load_features_clean                                        # noqa: E402

TRIAL = 123
OBSERVERS = [("016", "metal"), ("001", "glass")]
N_FRAMES = 6
INK, GRAY = "#222222", "#777777"
C_METAL, C_GLASS = "#1f3a6e", "#b9c9e3"          # the dark and light blue of the figures
FONT = "Helvetica Neue"
plt.rcParams.update({"font.family": [FONT, "Arial", "DejaVu Sans"], "font.size": 11})


def frames_of(pid, n):
    """n object-plus-surround patches, evenly spaced in time, from the Color frames."""
    d = next((IMGS / f"participant_{pid}" / "Transparent").glob(f"{TRIAL}_*"))
    fr = frame_paths(d)
    out = []
    for i in np.linspace(0, len(fr) - 1, n).round().astype(int):
        rgb, _ = context_crop(str(fr[i]).replace("/Transparent/", "/Color/"), fr[i], margin=0.9, size=600)
        out.append((rgb, (float(fr[i].stem) - float(fr[0].stem)) / 1000))
    return out


# the real statistics (z-scored within observer, as the model sees them) and the real predicted metallicity
F = load_features_clean(MAIN_FEATURES)
cols = [c for c in F.columns if c not in ("participant", "trial")]
Z = F.groupby("participant")[cols].transform(lambda s: (s - s.mean()) / (s.std() + 1e-12))
S = pd.read_csv(RESULTS / "models" / f"{MAIN_FEATURES}_C0.1_scores.csv", dtype={"participant": str})
rows = []
for pid, resp in OBSERVERS:
    z = Z[(F.participant == pid) & (F.trial == TRIAL)].values[0]
    s = float(S[(S.participant == pid) & (S.trial == TRIAL)].score_wt.iloc[0])
    rows.append(dict(pid=pid, resp=resp, z=z, s=s, frames=frames_of(pid, N_FRAMES)))

W, H = 16.0, 5.5                       # inches; 2400 x 825 px at 150 dpi
fig = plt.figure(figsize=(W, H), facecolor="white")
tile, gap = 1.28, 0.10                 # frame tiles, inches
left_x, top_y = 0.35, 0.62            # of the first tile, from the left / from the top
row_h = 2.45
ax_in = lambda x, y, w, h: fig.add_axes([x / W, 1 - (y + h) / H, w / W, h / H])

for r, row in enumerate(rows):
    y0 = top_y + r * row_h
    fig.text(left_x / W, 1 - (y0 - 0.28) / H, f'Observer {int(row["pid"])}  said "{row["resp"]}"', ha="left", va="center",
             fontsize=15, fontweight="bold", color=C_METAL if row["resp"] == "metal" else INK)
    fig.text((left_x + 3.0) / W, 1 - (y0 - 0.28) / H, "one trial of free exploration in virtual reality (head and hand), first-person view" if r == 0 else
             "the same object, same environment, a different observer, a different video", ha="left", va="center", fontsize=11, color=GRAY)
    for k, (rgb, t) in enumerate(row["frames"]):
        ax = ax_in(left_x + k * (tile + gap), y0, tile, tile)
        ax.imshow(rgb); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_edgecolor("#dddddd"); sp.set_linewidth(0.8)
        ax.set_title(f"{t:.1f} s", fontsize=9, color=GRAY, pad=3)
    # the 116 statistics of the video, as a strip of bars
    bx = left_x + N_FRAMES * (tile + gap) + 0.5
    axb = ax_in(bx, y0 + 0.1, 2.6, tile - 0.2)
    z = row["z"]
    axb.bar(np.arange(len(z)), z, width=1.0, color=C_METAL if row["resp"] == "metal" else "#5b7bb5", linewidth=0)
    axb.axhline(0, color=INK, lw=0.6); axb.set_xlim(-1, len(z)); axb.set_ylim(-3.2, 3.2)
    axb.set_xticks([1, 50, 100, 116]); axb.set_yticks([]); axb.tick_params(labelsize=8, colors=GRAY, length=2)
    for sp in ("top", "right", "left"): axb.spines[sp].set_visible(False)
    axb.spines["bottom"].set_color(GRAY)
    axb.set_title("116 statistics of this video (z-scored)" if r == 0 else "116 statistics of this video", fontsize=10, color=INK, pad=4)
    axb.set_xlabel("image and motion statistic", fontsize=8.5, color=GRAY, labelpad=1)
    # arrow from the frames to the bars
    ax_arrow = ax_in(bx - 0.5, y0, 0.5, tile); ax_arrow.axis("off"); ax_arrow.set_xlim(0, 1); ax_arrow.set_ylim(0, 1)
    ax_arrow.add_patch(FancyArrowPatch((0.1, 0.5), (0.85, 0.5), arrowstyle="-|>", mutation_scale=16, color=GRAY, lw=1.4))

# the model: a logistic regression fitted to the other observers' videos, the two videos on its curve
mx = left_x + N_FRAMES * (tile + gap) + 0.5 + 2.6 + 0.9
axm = ax_in(mx + 0.45, top_y + 0.35, W - mx - 0.45 - 0.4, row_h * 2 - 1.3)
ss = np.linspace(-6, 6, 300)
axm.plot(ss, 1 / (1 + np.exp(-ss)), color=INK, lw=2)
axm.axhline(0.5, color="#dddddd", lw=0.8, zorder=0)
for row in rows:
    s = row["s"]; p = 1 / (1 + np.exp(-s))
    c = C_METAL if row["resp"] == "metal" else C_GLASS
    axm.plot(s, p, "o", ms=15, mfc=c, mec=INK, mew=1.4, zorder=5)
axm.set_xlim(-6.2, 6.2); axm.set_ylim(-0.04, 1.04)
axm.set_xticks([-6, -3, 0, 3, 6]); axm.set_yticks([0, 0.5, 1]); axm.tick_params(labelsize=9, colors=GRAY, length=2)
for sp in ("top", "right"): axm.spines[sp].set_visible(False)
for sp in ("left", "bottom"): axm.spines[sp].set_color(GRAY)
axm.set_xlabel("predicted metallicity  s = w · x + b", fontsize=11, color=INK)
axm.set_ylabel('P("metal")', fontsize=11, color=INK)
axm.set_title("Logistic regression on the statistics,\nfitted to the other observers' videos", fontsize=11, color=INK, pad=8)
# arrows from the two bar strips into the model panel
for r in range(2):
    y0 = top_y + r * row_h
    ax_arrow = ax_in(mx - 0.9, y0, 0.9, tile); ax_arrow.axis("off"); ax_arrow.set_xlim(0, 1); ax_arrow.set_ylim(0, 1)
    ax_arrow.add_patch(FancyArrowPatch((0.1, 0.5), (0.9, 0.5), arrowstyle="-|>", mutation_scale=16, color=GRAY, lw=1.4))

fig.text(0.5, 0.05, "Does the video an observer generated, rather than the stimulus, explain that observer's response?  "
         "Trial 123 of Nomachi et al. (2026): bumpy object, physical metallicity 0.5, Sky environment.",
         ha="center", va="center", fontsize=10.5, color=GRAY)
fig.savefig(ROOT / "assets" / "thumbnail.png", dpi=150)
print("wrote", ROOT / "assets" / "thumbnail.png")
