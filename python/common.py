"""Shared paths and helpers of the Python pipeline.

Layout (all paths relative to the repository root):
  data/imgs/participant_XXX/<Color|Transparent|Cropped>/<trial folder>/<time_ms>.png
                           the first-person frames (not distributed, 200 GB)
  data/static/             the single frame of each trial without exploration (not distributed)
  data/trials.csv          one row per observer x trial (build_trial_table.py)
  results/features/        per-trial statistics, one CSV per participant and feature set
  results/models/          model outputs (evaluate.py and the analysis scripts)

Trial folders are named <trial ID>_<environment>_<shape>_<physical metallicity>_<response>,
e.g. 123_Sky_Bumpy_0.50_1; the trial ID gives the condition (modality()).
"""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
IMGS = ROOT / "data" / "imgs"
STATIC = ROOT / "data" / "static"   # the static block (trial IDs 1-40): one frame per trial, see import_static_block.py
RESULTS = ROOT / "results"
FEATURES = RESULTS / "features"

MAIN_FEATURES = "object_relational_flow"   # the main model: object (63) + relational (37) + motion (16) statistics = 116

TRIAL_RE = re.compile(r"^(\d+)_(Check|Gray|Sky|Studio)_(Bumpy|Smooth)_([01]\.\d\d)_([01])$")


def modality(trial_id: int) -> str:
    """The condition of a trial from its ID: hand only (41-80), head only (81-120),
    head and hand (121-160); 1-40 is the condition without exploration."""
    if 41 <= trial_id <= 80:
        return "hand"
    if 81 <= trial_id <= 120:
        return "head"
    if 121 <= trial_id <= 160:
        return "both"
    return "static"


def parse_trial_dir(name: str):
    m = TRIAL_RE.match(name)
    if m is None:
        return None
    t = int(m.group(1))
    return dict(trial=t, env=m.group(2), geom=m.group(3), metal=float(m.group(4)),
                resp=int(m.group(5)), modality=modality(t))


def window_mask(t_ms, window):
    """Which frames (time stamps t_ms, sorted) belong to a time window: 'all', or 'pre<A>to<B>s',
    the frames from B to A seconds before the response (e.g. pre2to4s), none when the video is
    shorter than B."""
    import numpy as np
    t = np.asarray(t_ms, float); n = len(t)
    m = re.match(r"^pre(\d+(?:\.\d+)?)to(\d+(?:\.\d+)?)s$", window)
    if m:
        a, b = float(m.group(1)) * 1000, float(m.group(2)) * 1000
        back = t[-1] - t                                        # time before the response
        if n == 0 or t[-1] - t[0] < b:                          # the video does not cover the window
            return np.zeros(n, bool)
        return (back >= a) & (back < b) if a > 0 else (back <= b)
    if window != "all":
        raise ValueError(window)
    return np.ones(n, bool)


def participants():
    """Participant IDs ('001' ...), from the frame folders when present, else from the trial table."""
    if IMGS.is_dir():
        return sorted(p.name[-3:] for p in IMGS.glob("participant_*"))
    import pandas as pd
    return sorted(pd.read_csv(ROOT / "data" / "trials.csv", dtype={"participant": str}).participant.unique())


# ---------------------------------------------------------------------------
# Object-plus-surround crops.  Glass transmits and metal reflects the
# environment, so a patch that includes a little of the background around the
# object is the smallest input in which the two can be told apart.
# ---------------------------------------------------------------------------
CROP_MARGIN = 0.25   # surround on each side, as a fraction of the object's bounding box
CROP_SIZE = 224


def object_bbox(alpha):
    """Bounding box (x0, y0, x1, y1) of the object from the alpha mask, or None."""
    import numpy as np
    ys, xs = np.where(alpha > 0)
    if len(xs) < 50:
        return None
    return xs.min(), ys.min(), xs.max() + 1, ys.max() + 1


def context_crop(color_path, transparent_path, margin=CROP_MARGIN, size=CROP_SIZE):
    """Square crop of the first-person view centred on the object, with a margin
    of `margin` x the object's size on each side, resized to `size` x `size`.
    Returns (rgb uint8 [size,size,3], object mask bool [size,size]) or None."""
    import numpy as np
    from PIL import Image
    a = np.asarray(Image.open(transparent_path).split()[-1])
    bb = object_bbox(a)
    if bb is None:
        return None
    x0, y0, x1, y1 = bb
    side = int(np.ceil(max(x1 - x0, y1 - y0) * (1 + 2 * margin)))
    H, W = a.shape
    side = min(side, H, W)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    L = int(round(min(max(cx - side / 2, 0), W - side)))
    T = int(round(min(max(cy - side / 2, 0), H - side)))
    box = (L, T, L + side, T + side)
    rgb = Image.open(color_path).convert("RGB").crop(box).resize((size, size), Image.BILINEAR)
    mask = Image.fromarray(a).crop(box).resize((size, size), Image.BILINEAR)
    return np.asarray(rgb), np.asarray(mask) > 127


def trial_dirs(pid, folder="Transparent", root=None):
    """Sorted (trial_dir, info) pairs of one participant."""
    out = []
    for d in sorted(((root or IMGS) / f"participant_{pid}" / folder).iterdir()):
        info = parse_trial_dir(d.name)
        if info is not None:
            out.append((d, info))
    return out


def frame_paths(tdir):
    return sorted(tdir.glob("*.png"), key=lambda p: float(p.stem))
