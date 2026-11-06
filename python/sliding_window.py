"""When the information that separates the observers arises: the disagreement test
on the statistics of a window of fixed length slid back from the response.

The windows (extract_time_windows.py) are 2 s long and end 0, 1, ... 6 s before
the response.  Every window holds the same amount of video, and only the videos
that cover the whole window enter it, so the number of videos falls as the
window moves away from the response (n_videos, n_pairs).  For the comparison of
the last window with the one before it, the last window is also tested on the
videos of the window before it only, those of 3 s or longer (min_duration_s = 3).

For each window the model is fitted as the main model (evaluate.py: its 116
statistics, z-scored within observer, L2 logistic regression with C = 0.1) and the
disagreement test is scored leave-one-stimulus-out with its permutation null (the
responses shuffled within each trial, 300 times).  The computation is written out
here instead of calling evaluate(), whose other measures (the per-observer
cross-validation, the predictor from the other observers) are not defined for
the few videos of the early windows; on the whole videos it returns the numbers
of evaluate() (python sliding_window.py --check).  The test is also computed per
observer (the pairs that contain the observer; observers with fewer than 20
such pairs in a window are left out), for the error bars.

Writes results/models/sliding_window.csv              one row per window and set of videos (min_duration_s: 0 = all)
       results/models/sliding_window_by_observer.csv  observer x window (disagreement test); the column of the
                                                      last window on the videos of 3 s or longer ends in _min3
"""
import re
import sys
from multiprocessing import Pool
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from common import RESULTS, MAIN_FEATURES
from evaluate import T, load_features, load_features_clean
from extract_time_windows import SLIDING

MIN_PAIRS = 20                                                 # per observer and window, for the per-observer AUC
DUR = T.assign(dur=(T.t_last_ms - T.t_first_ms) / 1000)[["participant", "trial", "dur"]]


def disagreement(name, min_dur=0, C=0.1, n_perm=300, seed=0):
    """The disagreement test of a feature set, on the statistics of the main model (a few motion summaries are
    undefined for the shortest videos and are not in it) and the videos of at least min_dur seconds: a dict of
    the summary and the per-observer AUCs."""
    main = set(load_features_clean(MAIN_FEATURES).columns)
    F = load_features(name)
    cols = [c for c in F.columns if c not in ("participant", "trial")]
    F[cols] = F[cols].replace([np.inf, -np.inf], np.nan)
    good = [c for c in cols if c in main and F[c].notna().mean() > 0.99 and F[c].std() > 0]
    F = F.dropna(subset=good).merge(T[["participant", "trial", "resp", "metal", "env", "geom"]],
                                    on=["participant", "trial"]).merge(DUR, on=["participant", "trial"])
    F = F[F.dur >= min_dur].reset_index(drop=True)
    Z = F.groupby("participant")[good].transform(lambda s: (s - s.mean()) / (s.std() + 1e-12)).fillna(0).values
    y, pids, trials = F.resp.values.copy(), F.participant.values, F.trial.values
    by_trial = {t: np.where(trials == t)[0] for t in np.unique(trials)}
    conds = (F.env + "_" + F.geom + "_" + F.metal.astype(str)).values     # the 40 stimuli
    rng = np.random.default_rng(seed)

    pred = np.zeros(len(y))                                    # leave-one-stimulus-out scores
    for c in np.unique(conds):
        te = conds == c
        m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000)).fit(Z[~te], y[~te])
        pred[te] = m.decision_function(Z[te])

    def within_trial(yy):
        ok = tot = 0.0
        for idx in by_trial.values():
            pos, neg = idx[yy[idx] == 1], idx[yy[idx] == 0]
            for i in pos:
                ok += np.sum(pred[i] > pred[neg]) + 0.5 * np.sum(pred[i] == pred[neg]); tot += len(neg)
        return ok / tot, int(tot)

    auc, n_pairs = within_trial(y)
    null = []
    for _ in range(n_perm):
        yp = y.copy()
        for idx in by_trial.values():
            yp[idx] = rng.permutation(yp[idx])
        null.append(within_trial(yp)[0])
    null = np.array(null)

    obs = {}                                                   # per observer: the pairs that contain the observer
    for p in np.unique(pids):
        ok = tot = 0.0
        for idx in by_trial.values():
            me = idx[pids[idx] == p]
            if len(me) != 1:
                continue
            ot = idx[(pids[idx] != p) & (y[idx] != y[me[0]])]
            d = (pred[me[0]] - pred[ot]) * (1 if y[me[0]] == 1 else -1)
            ok += np.sum(d > 0) + 0.5 * np.sum(d == 0); tot += len(ot)
        if tot >= MIN_PAIRS:
            obs[p] = ok / tot
    obs = pd.Series(obs, dtype=float)
    res = dict(feature_set=name, n_features=len(good), n_videos=len(y), n_pairs=n_pairs,
               within_trial_auc=auc, null_mean=null.mean(), null_sd=null.std(), z=(auc - null.mean()) / null.std(),
               p=(np.sum(null >= auc) + 1) / (n_perm + 1),
               obs_mean=obs.mean(), obs_se=obs.std() / np.sqrt(len(obs)), n_obs=len(obs))
    return res, obs


def run(job):
    window, min_dur = job
    a, b = map(float, re.match(r"pre(.+)to(.+)s$", window).groups())
    res, obs = disagreement(f"{MAIN_FEATURES}_{window}", min_dur)
    row = dict(window=window, start_s=-b, end_s=0.0 - a if a else 0.0, center_s=-(a + b) / 2, min_duration_s=min_dur, **res)
    print({k: (round(float(v), 3) if isinstance(v, (float, np.floating)) else v) for k, v in row.items()}, flush=True)
    return row, obs.rename(window if min_dur == 0 else f"{window}_min{min_dur:g}")


if __name__ == "__main__":
    if "--check" in sys.argv:                                  # against evaluate(): the whole videos
        res, obs = disagreement(MAIN_FEATURES)
        print({k: (round(float(v), 4) if isinstance(v, (float, np.floating)) else v) for k, v in res.items()})
        sys.exit()
    starts = {w: float(re.match(r"pre(.+)to(.+)s$", w).group(2)) for w in SLIDING}
    jobs = [(w, 0) for w in SLIDING] + [(SLIDING[0], int(starts[SLIDING[1]]))]   # and the last window on the videos of the window before it
    with Pool(4) as pool:
        out = pool.map(run, jobs)
    R = pd.DataFrame([r for r, _ in out]).sort_values(["min_duration_s", "center_s"])
    R.to_csv(RESULTS / "models" / "sliding_window.csv", index=False)
    W = pd.concat([w for _, w in out], axis=1); W.index.name = "participant"
    W.sort_index().to_csv(RESULTS / "models" / "sliding_window_by_observer.csv")
    print(R.drop(columns=["feature_set"]).round(3).to_string(index=False))
