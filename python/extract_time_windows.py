"""The statistics of every video from a window of fixed length (2 s) slid back
from the response in steps of 1 s: from 2 to 0 s before the response to from 8
to 6 s before it.  Only the videos that cover the whole window have statistics
for it.  For the analysis of when the information arises (sliding_window.py).

The object statistics are recomputed from the frames with the functions of
extract_image_stats.py (the same frame selection: the window, then at most 150
frames evenly spaced; per-frame results are cached within a trial, since the
windows overlap); the relational and motion statistics are re-summarized from
their per-frame files (extract_relational_stats.py, extract_flow_cues.py).

Usage: python extract_time_windows.py [participant ...]
Writes results/features/{image_stats_object,relational,flow}_pre<A>to<B>s/participant_XXX.csv (from B to A s before the response)
"""
import sys
from multiprocessing import Pool
import numpy as np
import pandas as pd
from common import FEATURES, participants, trial_dirs, frame_paths, window_mask
import extract_image_stats as eis
import extract_relational_stats as ers
import extract_flow_cues as efc

LENGTH, ENDS = 2, range(0, 7)                                 # the window: its length and its ends [s before the response]
SLIDING = [f"pre{e:g}to{e + LENGTH:g}s" for e in ENDS]
WINDOWS = SLIDING


def trial_windows(tdir):
    """The per-trial statistics of one video for every window: {window: dict or None}.
    The same computation as extract_image_stats.trial_features, sharing the per-frame work."""
    eis.CROP = "object"
    frames = frame_paths(tdir)
    t = [float(p.stem) for p in frames]
    cache = {}

    def frame(fp):
        if fp not in cache:
            rgb, obj = eis.load_patch(tdir, fp)
            cache[fp] = eis.frame_stats(rgb, obj)               # (statistics, luminance image, region) or (None, None, None)
        return cache[fp]

    out = {}
    for w in WINDOWS:
        sel = [f for f, k in zip(frames, window_mask(t, w)) if k]
        if len(sel) > eis.MAX_FRAMES:
            sel = [sel[i] for i in np.linspace(0, len(sel) - 1, eis.MAX_FRAMES).round().astype(int)]
        ts, prevY, prevM, diffs = [], None, None, []
        for fp in sel:
            s, Y, region = frame(fp)
            if s is None:
                continue
            if prevY is not None:                               # change from the previous sampled frame, inside the object in both
                m = region & prevM
                diffs.append(np.abs(Y - prevY)[m].mean() if m.sum() > 50 else np.nan)
            prevY, prevM = Y, region
            ts.append(s)
        if len(ts) < 5:
            out[w] = None; continue
        D = pd.DataFrame(ts)
        D["diff_energy"] = [np.nan] + diffs
        o = dict(n_frames_used=len(D), n_frames_total=len(frames))
        h = len(D) // 2
        for c in D.columns:
            v = D[c].values
            o[f"{c}_mean"] = np.nanmean(v); o[f"{c}_sd"] = np.nanstd(v); o[f"{c}_max"] = np.nanmax(v)
            o[f"{c}_late_minus_early"] = np.nanmean(v[h:]) - np.nanmean(v[:h])
        for c in ("Y_mean", "rms_contrast", "hl_frac"):
            v = D[c].values; v = v - v.mean()
            o[f"{c}_ac1"] = (v[:-1] * v[1:]).mean() / (v.var() + 1e-12)
        out[w] = o
    return out


def run_participant(pid):
    rows = {w: [] for w in WINDOWS}
    for d, info in trial_dirs(pid, "Cropped"):
        for w, f in trial_windows(d).items():
            if f is not None:
                rows[w].append(dict(participant=pid, trial=info["trial"], **f))
    for w in WINDOWS:
        out = FEATURES / f"image_stats_object_{w}"; out.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows[w]).to_csv(out / f"participant_{pid}.csv", index=False)
        ers.summarize(pid, w)                                   # relational and motion statistics: from the per-frame files
        efc.summarize(pid, w)
    print(pid, {w: len(r) for w, r in rows.items()}, flush=True)
    return pid


if __name__ == "__main__":
    import warnings; warnings.filterwarnings("ignore", category=RuntimeWarning)
    with Pool(6) as pool:
        pool.map(run_participant, sys.argv[1:] or participants())
