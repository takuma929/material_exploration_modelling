"""Optic-flow cues to shininess of Doerschner et al. (2011, Current Biology):
coverage change, divergence and 3D shape reliability, computed for every
pair of consecutive frames of every trial on the object crops, then
summarized over the same frame samples as the other statistics.

Per frame k (pair k -> k+STEP, and k -> k+2*STEP for the coverage change; STEP = 2
frames, about 34 ms), on the object enlarged to fill a 224 x 224 image:
  coverage          fraction of object points tracked from frame k to k+1
                    (pyramidal Lucas-Kanade with a forward-backward check)
  coverage_change   percent decrease of coverage from a 1-frame to a 2-frame distance
  div_frac          fraction of object pixels with |divergence| > 0.5 among those with
                    nonzero divergence (dense Farneback flow, 1-frame distance)
  div_mean_abs      mean |divergence| over the object (continuous variant)
  sfm_inlier_ratio  fraction of tracked vectors (magnitude > 0.25 SD) that are inliers of
                    a fundamental matrix fitted by RANSAC (Sampson error < 1 px):
                    consistency of the flow with a rigid 3-D motion
  flow_speed        mean magnitude of the tracked vectors [px per frame]

Of the six quantities, div_frac and sfm_inlier_ratio are undefined in 1 to 2% of
the videos (too few trackable points) and are dropped by evaluate.py's rule
against columns with more than 1% missing, which leaves 4 quantities x 4
summaries = 16 motion statistics in the main model.

Usage: python extract_flow_cues.py [participant ...]
  Without arguments, every participant whose per-frame file does not exist yet is
  processed, then the per-trial summaries are (re)written.
Writes results/features/flow_frames/participant_XXX.csv (one row per frame pair)
       results/features/flow/participant_XXX.csv
"""
import sys
from multiprocessing import Pool
import numpy as np
import pandas as pd
import cv2
from common import IMGS, FEATURES, participants, trial_dirs, frame_paths, window_mask

MAX_FRAMES = 150
FRAMES = FEATURES / "flow_frames"
QUANT = ["coverage", "coverage_change", "div_frac", "div_mean_abs", "sfm_inlier_ratio", "flow_speed"]
LK = dict(winSize=(15, 15), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
GRID = 6          # grid spacing of the tracked points [px]
DIV_THRESH = 0.5     # Doerschner et al. used 2 at their image scale and rotation speed; our frames are 34 ms apart
STEP = 2             # frame distance: every second sampled frame (about 34 ms), the coverage change uses 2 x STEP


def load_gray(path):
    im = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return im


def zoom_box(im, margin=0.15):
    """Bounding box of the object (non-black pixels) plus a margin, as a square."""
    ys, xs = np.where(im > 3)
    if len(xs) < 50:
        return None
    x0, y0, x1, y1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
    side = int(np.ceil(max(x1 - x0, y1 - y0) * (1 + 2 * margin)))
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    H, W = im.shape
    L = int(round(min(max(cx - side / 2, 0), max(W - side, 0)))); T = int(round(min(max(cy - side / 2, 0), max(H - side, 0))))
    return L, T, side


def zoom(im, box):
    L, T, side = box
    crop = np.zeros((side, side), im.dtype)
    sub = im[T:T + side, L:L + side]
    crop[:sub.shape[0], :sub.shape[1]] = sub
    return cv2.resize(crop, (224, 224), interpolation=cv2.INTER_LINEAR)


def track(g0, g1, pts):
    p1, st, _ = cv2.calcOpticalFlowPyrLK(g0, g1, pts, None, **LK)
    p0b, st2, _ = cv2.calcOpticalFlowPyrLK(g1, g0, p1, None, **LK)
    fb = np.linalg.norm(p0b - pts, axis=2).ravel()
    ok = (st.ravel() == 1) & (st2.ravel() == 1) & (fb < 1.0)
    return p1, ok


def pair_cues(g0, g1, g2, mask):
    """g0, g1, g2: consecutive object frames (uint8, 224 x 224, same zoom); mask: object pixels of g0."""
    ys, xs = np.where(mask)
    sel = (ys % GRID == 0) & (xs % GRID == 0)
    pts = np.stack([xs[sel], ys[sel]], axis=1).astype(np.float32).reshape(-1, 1, 2)
    if len(pts) < 20:
        return None
    p1, ok1 = track(g0, g1, pts)
    cov1 = ok1.mean()
    if g2 is not None:
        _, ok2 = track(g0, g2, pts)
        cov_change = 100 * (cov1 - ok2.mean()) / (cov1 + 1e-6)
    else:
        cov_change = np.nan
    v = (p1 - pts).reshape(-1, 2)[ok1]
    speed = np.linalg.norm(v, axis=1)
    out = dict(coverage=cov1, coverage_change=cov_change, flow_speed=speed.mean() if len(speed) else np.nan)
    # divergence of the dense flow over the object
    flow = cv2.calcOpticalFlowFarneback(g0, g1, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    du_dx = np.gradient(flow[..., 0], axis=1); dv_dy = np.gradient(flow[..., 1], axis=0)
    div = (du_dx + dv_dy)[mask]
    nz = div[np.abs(div) > 1e-6]
    out["div_frac"] = (np.abs(nz) > DIV_THRESH).mean() if len(nz) else np.nan
    out["div_mean_abs"] = np.abs(div).mean() if len(div) else np.nan
    # 3-D shape reliability: inliers of a fundamental matrix among the larger vectors
    if len(speed) >= 16:
        keep = speed > 0.25 * speed.std()
        a = pts.reshape(-1, 2)[ok1][keep]; b = p1.reshape(-1, 2)[ok1][keep]
        if len(a) >= 16:
            _, inl = cv2.findFundamentalMat(a, b, cv2.FM_RANSAC, 1.0, 0.99, 100)
            out["sfm_inlier_ratio"] = inl.ravel().mean() if inl is not None else np.nan
        else:
            out["sfm_inlier_ratio"] = np.nan
    else:
        out["sfm_inlier_ratio"] = np.nan
    return out


def frames_of_participant(pid):
    rows = []
    for d, info in trial_dirs(pid, "Cropped"):
        fr = frame_paths(d)
        ims = [load_gray(f) for f in fr]
        for k in range(len(fr) - STEP):
            box = zoom_box(ims[k])
            if box is None:
                continue
            g0 = zoom(ims[k], box); g1 = zoom(ims[k + STEP], box)
            g2 = zoom(ims[k + 2 * STEP], box) if k + 2 * STEP < len(fr) else None
            q = pair_cues(g0, g1, g2, g0 > 3)
            if q is None:
                continue
            rows.append(dict(participant=pid, trial=info["trial"], frame=k, t_ms=float(fr[k].stem), **q))
    FRAMES.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(FRAMES / f"participant_{pid}.csv", index=False)
    print(pid, len(rows), "frame pairs", flush=True)
    return pid


def select(D, window):
    D = D[window_mask(D.t_ms.values, window)]
    if len(D) > MAX_FRAMES:
        D = D.iloc[np.linspace(0, len(D) - 1, MAX_FRAMES).round().astype(int)]
    return D


def summarize(pid, window):
    F = pd.read_csv(FRAMES / f"participant_{pid}.csv", dtype={"participant": str})
    rows = []
    for trial, D in F.groupby("trial"):
        D = select(D.sort_values("frame"), window)
        if len(D) < 5:
            continue
        out = dict(participant=pid, trial=trial, n_frames_used=len(D))
        h = len(D) // 2
        for c in QUANT:
            v = D[c].values
            out[f"{c}_mean"] = np.nanmean(v); out[f"{c}_sd"] = np.nanstd(v); out[f"{c}_max"] = np.nanmax(v)
            out[f"{c}_late_minus_early"] = np.nanmean(v[h:]) - np.nanmean(v[:h])
        rows.append(out)
    out_dir = FEATURES / ("flow" if window == "all" else f"flow_{window}")
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_dir / f"participant_{pid}.csv", index=False)


if __name__ == "__main__":
    pids = sys.argv[1:] or participants()
    todo = [p for p in pids if not (FRAMES / f"participant_{p}.csv").exists()]
    with Pool(4) as pool:
        pool.map(frames_of_participant, todo)
    for pid in pids:
        summarize(pid, "all")                                   # the windows before the response: extract_time_windows.py
    print("summaries written", flush=True)
