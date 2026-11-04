"""Import the static block (trial IDs 1-40, no exploration) from the raw
export in data/trial1-40 into data/static.

The observer could not move in this block, so every frame of a trial is the
same image; only the viewing duration differs.  For each trial the script
keeps the first frame (Color and Transparent), writes a 224 x 224 object-only
crop made as the Cropped frames of the moving blocks (a fixed 143-px window
centred on the object, resized to 224) and records the duration from the
frame timestamps.

Usage: python import_static_block.py [raw_dir]
Writes data/static/participant_XXX/{Color,Transparent,Cropped}/<trial>/<t>.png
       data/static_trials.csv
"""
import shutil
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import label
from common import ROOT, STATIC, parse_trial_dir

RAW = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "trial1-40"
CANVAS = 143      # the common canvas of preprocessing/crop_transparent_objects.m (99th percentile of the object boxes)
SIZE = 224


def object_crop(transparent_png):
    """Object-only crop as in crop_transparent_objects.m: non-black pixels of
    the RGB channels, largest component, a CANVAS x CANVAS window centred on
    its bounding box, zero padded, resized to SIZE x SIZE."""
    im = np.asarray(Image.open(transparent_png))
    rgb = im[..., :3]
    bw = rgb.astype(np.float32).mean(-1) / 255 > 0.02
    lab, n = label(bw)
    if n == 0:
        return None
    k = 1 + np.argmax(np.bincount(lab.ravel())[1:])
    ys, xs = np.where(lab == k)
    cx, cy = (xs.min() + xs.max() + 1) / 2, (ys.min() + ys.max() + 1) / 2
    x1, y1 = int(round(cx - CANVAS / 2)), int(round(cy - CANVAS / 2))
    out = np.zeros((CANVAS, CANVAS, 3), np.uint8)
    H, W = bw.shape
    sx1, sy1, sx2, sy2 = max(0, x1), max(0, y1), min(W, x1 + CANVAS), min(H, y1 + CANVAS)
    out[sy1 - y1:sy2 - y1, sx1 - x1:sx2 - x1] = rgb[sy1:sy2, sx1:sx2]
    return Image.fromarray(out).resize((SIZE, SIZE), Image.BICUBIC)


rows = []
for pdir in sorted(RAW.glob("participant_*"), key=lambda p: int(p.name.split("_")[1])):
    pid = f"{int(pdir.name.split('_')[1]):03d}"
    for tdir in sorted((pdir / "Transparent").iterdir()):
        info = parse_trial_dir(tdir.name)
        if info is None or not tdir.is_dir():
            continue
        frames = sorted(tdir.glob("*.png"), key=lambda p: float(p.stem))
        cdir = pdir / "Color" / tdir.name
        cframes = sorted(cdir.glob("*.png"), key=lambda p: float(p.stem))
        assert len(frames) == len(cframes) and frames[0].name == cframes[0].name, (pid, tdir.name)
        first = frames[0]
        dst = STATIC / f"participant_{pid}"
        for variant, src in (("Transparent", first), ("Color", cdir / first.name)):
            d = dst / variant / tdir.name
            d.mkdir(parents=True, exist_ok=True)
            if not (d / first.name).exists():
                shutil.copy2(src, d / first.name)
        crop = object_crop(first)
        d = dst / "Cropped" / tdir.name
        d.mkdir(parents=True, exist_ok=True)
        crop.save(d / first.name)
        t = [float(p.stem) for p in frames]
        rows.append(dict(participant=pid, folder=tdir.name, **info, n_frames=len(frames),
                         t_first_ms=t[0], t_last_ms=t[-1], dur_s=(t[-1] - t[0]) / 1000, frame_kept=first.name))
    print(pid, sum(r["participant"] == pid for r in rows), "trials", flush=True)
df = pd.DataFrame(rows).sort_values(["participant", "trial"]).reset_index(drop=True)
df.to_csv(ROOT / "data" / "static_trials.csv", index=False)
print(df.groupby("participant").size().describe())
print("duration (s): median %.2f, min %.2f, max %.2f" % (df.dur_s.median(), df.dur_s.min(), df.dur_s.max()))
