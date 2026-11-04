"""Relational statistics: the object against its surround, per frame, for
every frame of every trial, then summarized over the same frame samples and
time windows as the object statistics.

Per frame (common.context_crop patch, object mask = the rendered alpha):
  bg_Y_mean, bg_Y_sd, bg_Y_max, bg_rms_contrast, bg_chroma_mean   the surround
  obj_bg_lum_ratio       log of object mean luminance over surround mean luminance
  obj_bg_contrast_ratio  log of object RMS contrast over surround RMS contrast
  obj_bg_max_ratio       log of object maximum luminance over surround maximum
  obj_bg_chroma_ratio    log of (object chroma + 1) over (surround chroma + 1)
Glass transmits and metal reflects the surroundings, so these are the
statistics in which the two can differ even when the object's own statistics
do not.

Usage: python extract_relational_stats.py [participant ...]
  Without arguments, every participant whose per-frame file does not exist yet is
  processed, then the per-trial summaries of every window are (re)written.
Writes results/features/relational_frames/participant_XXX.csv  (one row per frame)
       results/features/relational[_first2s|_early_half|_late_half]/participant_XXX.csv
"""
import sys
from multiprocessing import Pool
import numpy as np
import pandas as pd
from common import FEATURES, participants, trial_dirs, frame_paths, context_crop

MAX_FRAMES = 150
FRAMES = FEATURES / "relational_frames"
QUANT = ["bg_Y_mean", "bg_Y_sd", "bg_Y_max", "bg_rms_contrast", "bg_chroma_mean",
         "obj_bg_lum_ratio", "obj_bg_contrast_ratio", "obj_bg_max_ratio", "obj_bg_chroma_ratio"]


def srgb_to_lin(x):
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def frame_quantities(rgb, obj):
    lin = srgb_to_lin(rgb)
    Y = 0.2126 * lin[..., 0] + 0.7152 * lin[..., 1] + 0.0722 * lin[..., 2]
    X = 0.4124 * lin[..., 0] + 0.3576 * lin[..., 1] + 0.1805 * lin[..., 2]
    Zc = 0.0193 * lin[..., 0] + 0.1192 * lin[..., 1] + 0.9505 * lin[..., 2]
    f = lambda t: np.where(t > 0.008856, np.cbrt(t), 7.787 * t + 16 / 116)
    fx, fy, fz = f(X / 0.9505), f(Y), f(Zc / 1.089)
    chroma = np.sqrt((500 * (fx - fy)) ** 2 + (200 * (fy - fz)) ** 2)
    bg = ~obj
    if obj.sum() < 50 or bg.sum() < 50:
        return None
    yo, yb, co, cb = Y[obj], Y[bg], chroma[obj], chroma[bg]
    e = 1e-3
    rms = lambda y: y.std() / (y.mean() + e)
    return dict(bg_Y_mean=yb.mean(), bg_Y_sd=yb.std(), bg_Y_max=yb.max(), bg_rms_contrast=rms(yb), bg_chroma_mean=cb.mean(),
                obj_bg_lum_ratio=np.log((yo.mean() + e) / (yb.mean() + e)),
                obj_bg_contrast_ratio=np.log((rms(yo) + e) / (rms(yb) + e)),
                obj_bg_max_ratio=np.log((yo.max() + e) / (yb.max() + e)),
                obj_bg_chroma_ratio=np.log((co.mean() + 1) / (cb.mean() + 1)))


def frames_of_participant(pid):
    rows = []
    for d, info in trial_dirs(pid):
        for k, fp in enumerate(frame_paths(d)):
            r = context_crop(str(fp).replace("/Transparent/", "/Color/"), fp)
            if r is None:
                continue
            q = frame_quantities(r[0].astype(np.float32) / 255, r[1])
            if q is None:
                continue
            rows.append(dict(participant=pid, trial=info["trial"], frame=k, t_ms=float(fp.stem), **q))
    FRAMES.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(FRAMES / f"participant_{pid}.csv", index=False)
    print(pid, len(rows), "frames", flush=True)
    return pid


def select(D, window):
    """Same frame selection as extract_image_stats: window, then at most MAX_FRAMES evenly spaced."""
    t = D.t_ms.values
    if window == "first2s":
        D = D[t - t[0] <= 2000]
    elif window == "late_half":
        D = D.iloc[len(D) // 2:]
    elif window == "early_half":
        D = D.iloc[: max(5, len(D) // 2)]
    if len(D) > MAX_FRAMES:
        D = D.iloc[np.linspace(0, len(D) - 1, MAX_FRAMES).round().astype(int)]
    return D


def summarize(pid, window, shuffle=None):
    """Window summaries of the per-frame series; with shuffle=SEED the frames of
    every trial are put in a random order first (a control for the temporal structure)."""
    F = pd.read_csv(FRAMES / f"participant_{pid}.csv", dtype={"participant": str})
    rows = []
    for trial, D in F.groupby("trial"):
        D = select(D.sort_values("frame"), window)
        if len(D) < 5:
            continue
        if shuffle is not None:
            rs = np.random.RandomState(shuffle * 100003 + int(trial) * 7)
            D = D.iloc[rs.permutation(len(D))]
        out = dict(participant=pid, trial=trial, n_frames_used=len(D))
        h = len(D) // 2
        for c in QUANT:
            v = D[c].values
            out[f"{c}_mean"] = v.mean(); out[f"{c}_sd"] = v.std(); out[f"{c}_max"] = v.max()
            out[f"{c}_late_minus_early"] = v[h:].mean() - v[:h].mean()
        v = D["obj_bg_lum_ratio"].values; v = v - v.mean()
        out["obj_bg_lum_ratio_ac1"] = (v[:-1] * v[1:]).mean() / (v.var() + 1e-12)
        rows.append(out)
    out_dir = FEATURES / (("relational" if window == "all" else f"relational_{window}") + ("" if shuffle is None else f"_shuffle{shuffle}"))
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_dir / f"participant_{pid}.csv", index=False)


if __name__ == "__main__":
    if "--shuffle" in sys.argv:
        i = sys.argv.index("--shuffle"); seed = int(sys.argv[i + 1]); del sys.argv[i:i + 2]
        for pid in (sys.argv[1:] or participants()):
            summarize(pid, "all", shuffle=seed)
        print("shuffled relational summaries written", seed, flush=True); sys.exit()
    pids = sys.argv[1:] or participants()
    todo = [p for p in pids if not (FRAMES / f"participant_{p}.csv").exists()]
    with Pool(8) as pool:
        pool.map(frames_of_participant, todo)
    for pid in pids:
        for window in ("all", "first2s", "late_half", "early_half"):
            summarize(pid, window)
    print("summaries written", flush=True)
