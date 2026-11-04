"""Hand-designed image statistics of the frames each observer saw.

Per frame, in object mode (the main analysis) over the object's own pixels
with a 2-px band inside the silhouette removed, so that edge and frame-
difference energies measure the surface and not the contour against the
background; in context mode over a 224x224 patch that includes a margin of
surround (see common.context_crop).  Statistics: luminance distribution, RMS
contrast, highlight coverage, edge energy, Lab chroma, object-vs-surround
luminance and contrast (context mode), frame-difference energy.  Per trial: mean, sd, max,
first-vs-last-half change of each time series and lag-1 autocorrelations.

Usage: python extract_image_stats.py [--window all|first2s|early_half|late_half]
                                     [--crop context|object] [--shuffle SEED] [participant ...]
Writes results/features/image_stats[_object][_<window>][_shuffleSEED]/participant_XXX.csv
The paper uses --crop object (the object statistics, 63 per trial), once per
window.  With --shuffle the sampled frames of every trial are put in a random
order before the summaries are computed (a control for the temporal structure,
not reported).  Context mode computes the same statistics over the whole patch
(object and surround together), for comparison only.
"""
import sys
from multiprocessing import Pool
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import binary_erosion
from common import IMGS, FEATURES, participants, trial_dirs, frame_paths, context_crop

MAX_FRAMES = 150
CONTOUR_PX = 2          # band inside the silhouette excluded in object mode
WINDOW, CROP, SHUFFLE = "all", "context", None
if "--shuffle" in sys.argv:
    i = sys.argv.index("--shuffle"); SHUFFLE = int(sys.argv[i + 1]); del sys.argv[i:i + 2]
if "--window" in sys.argv:
    i = sys.argv.index("--window"); WINDOW = sys.argv[i + 1]; del sys.argv[i:i + 2]
if "--crop" in sys.argv:
    i = sys.argv.index("--crop"); CROP = sys.argv[i + 1]; del sys.argv[i:i + 2]
def out_dir():
    return FEATURES / ("image_stats" + ("_object" if CROP == "object" else "") + ("" if WINDOW == "all" else f"_{WINDOW}")
                       + ("" if SHUFFLE is None else f"_shuffle{SHUFFLE}"))
OUT = out_dir()
OUT.mkdir(parents=True, exist_ok=True)


def select_window(frames):
    t = np.array([float(p.stem) for p in frames])
    if WINDOW == "first2s":
        return [f for f, tt in zip(frames, t) if tt - t[0] <= 2000]
    if WINDOW == "early_half":
        return frames[: max(5, len(frames) // 2)]
    if WINDOW == "late_half":
        return frames[len(frames) // 2:]
    return frames


def srgb_to_lin(x):
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def frame_stats(rgb, obj):
    """rgb float [0,1]; obj = object mask.  Statistics are computed over the
    whole patch (object + surround) in context mode, over the object in object mode."""
    lin = srgb_to_lin(rgb)
    Y = 0.2126 * lin[..., 0] + 0.7152 * lin[..., 1] + 0.0722 * lin[..., 2]
    region = np.ones_like(obj) if CROP == "context" else binary_erosion(obj, iterations=CONTOUR_PX)
    if obj.sum() < 50 or region.sum() < 50:
        return None, None, None
    y = Y[region]
    gx = np.diff(Y, axis=1)[:-1, :]; gy = np.diff(Y, axis=0)[:, :-1]
    m2 = region[:-1, :-1] & region[:-1, 1:] & region[1:, :-1]   # every pixel of the stencil inside
    X = 0.4124 * lin[..., 0] + 0.3576 * lin[..., 1] + 0.1805 * lin[..., 2]
    Zc = 0.0193 * lin[..., 0] + 0.1192 * lin[..., 1] + 0.9505 * lin[..., 2]
    f = lambda t: np.where(t > 0.008856, np.cbrt(t), 7.787 * t + 16 / 116)
    fx, fy, fz = f(X / 0.9505), f(Y), f(Zc / 1.089)
    a, b = 500 * (fx - fy), 200 * (fy - fz)
    chroma = np.sqrt(a ** 2 + b ** 2)[region]
    hl = y >= np.quantile(y, 0.99)
    s = dict(
        area=obj.mean(), Y_mean=y.mean(), Y_sd=y.std(), Y_med=np.median(y), Y_max=y.max(),
        Y_skew=((y - y.mean()) ** 3).mean() / (y.std() ** 3 + 1e-12),
        Y_kurt=((y - y.mean()) ** 4).mean() / (y.std() ** 4 + 1e-12),
        rms_contrast=y.std() / (y.mean() + 1e-12),
        hl_frac=(y > 0.5 * y.max()).mean(), hl_contrast=y[hl].std() / (y[hl].mean() + 1e-12),
        edge_energy=np.sqrt(gx[m2] ** 2 + gy[m2] ** 2).mean(),
        dark_frac=(y < 0.2 * y.mean()).mean(),
        chroma_mean=chroma.mean(), chroma_sd=chroma.std(),
    )
    if CROP == "context":
        yo, yb = Y[obj], Y[~obj]
        if len(yb) >= 50:
            s["obj_bg_lum_ratio"] = np.log((yo.mean() + 1e-3) / (yb.mean() + 1e-3))
            s["obj_bg_contrast_ratio"] = np.log((yo.std() / (yo.mean() + 1e-3) + 1e-3) / (yb.std() / (yb.mean() + 1e-3) + 1e-3))
        else:
            s["obj_bg_lum_ratio"] = np.nan; s["obj_bg_contrast_ratio"] = np.nan
    return s, Y, region


def load_patch(tdir, fp):
    if CROP == "context":
        r = context_crop(str(fp).replace("/Transparent/", "/Color/"), fp)
        if r is None:
            return None, None
        rgb, obj = r
        return rgb.astype(np.float32) / 255, obj
    rgb = np.asarray(Image.open(fp).convert("RGB"), dtype=np.float32) / 255
    Y = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    return rgb, Y > 0.01


def trial_features(tdir):
    frames = select_window(frame_paths(tdir))
    if len(frames) > MAX_FRAMES:
        frames = [frames[i] for i in np.linspace(0, len(frames) - 1, MAX_FRAMES).round().astype(int)]
    if SHUFFLE is not None:
        rs = np.random.RandomState(SHUFFLE * 100003 + int(tdir.name.split('_')[0]) * 7 + (1 if '_1' in tdir.name[-2:] else 0))
        frames = [frames[i] for i in rs.permutation(len(frames))]
    ts, prevY, prevM, diffs = [], None, None, []
    for fp in frames:
        rgb, obj = load_patch(tdir, fp)
        if rgb is None:
            continue
        s, Y, region = frame_stats(rgb, obj)
        if s is None:
            continue
        if prevY is not None:
            m = region & prevM
            diffs.append(np.abs(Y - prevY)[m].mean() if m.sum() > 50 else np.nan)
        prevY, prevM = Y, region
        ts.append(s)
    if len(ts) < 5:
        return None
    D = pd.DataFrame(ts)
    D["diff_energy"] = [np.nan] + diffs
    out = dict(n_frames_used=len(D), n_frames_total=len(frame_paths(tdir)))
    h = len(D) // 2
    for c in D.columns:
        v = D[c].values
        out[f"{c}_mean"] = np.nanmean(v); out[f"{c}_sd"] = np.nanstd(v); out[f"{c}_max"] = np.nanmax(v)
        out[f"{c}_late_minus_early"] = np.nanmean(v[h:]) - np.nanmean(v[:h])
    for c in ("Y_mean", "rms_contrast", "hl_frac"):
        v = D[c].values; v = v - v.mean()
        out[f"{c}_ac1"] = (v[:-1] * v[1:]).mean() / (v.var() + 1e-12)
    return out


def _init(window, crop, shuffle):
    """Pool workers are spawned on macOS and do not inherit the parsed options."""
    global WINDOW, CROP, SHUFFLE, OUT
    WINDOW, CROP, SHUFFLE = window, crop, shuffle
    OUT = out_dir()


def run_participant(pid):
    folder = "Transparent" if CROP == "context" else "Cropped"
    rows = []
    for d, info in trial_dirs(pid, folder):
        f = trial_features(d)
        if f is None:
            print("skip", pid, d.name, flush=True); continue
        rows.append(dict(participant=pid, trial=info["trial"], **f))
    pd.DataFrame(rows).to_csv(OUT / f"participant_{pid}.csv", index=False)
    print(pid, len(rows), "trials", flush=True)
    return pid


if __name__ == "__main__":
    pids = sys.argv[1:] or participants()
    with Pool(8, initializer=_init, initargs=(WINDOW, CROP, SHUFFLE)) as pool:
        pool.map(run_participant, pids)
