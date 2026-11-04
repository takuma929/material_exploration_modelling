"""Is the observer-specific information just the viewing duration, and which
statistics carry it?

Duration control: the main model scored leave-one-stimulus-out as in
evaluate.py, (i) as it is, (ii) with the log of the number of frames partialled
out of every statistic, and (iii) the log duration alone, each in the
disagreement test, over all pairs, by condition and per observer.
Single statistics: the disagreement test on one statistic at a time (z-scored
within observer).  Ablations: the disagreement test on subsets of the statistics.

Writes results/models/duration_control.csv, duration_control_by_observer.csv,
       within_trial_features.csv and feature_ablations.csv
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from common import ROOT, RESULTS, FEATURES, MAIN_FEATURES
from evaluate import load_features_clean, T, PIDS

F = load_features_clean(MAIN_FEATURES)
dur = pd.concat([pd.read_csv(f, dtype={"participant": str})[["participant", "trial", "n_frames_total"]]
                 for f in sorted((FEATURES / "image_stats_object").glob("participant_*.csv"))])
F = F.merge(dur, on=["participant", "trial"]).merge(T[["participant", "trial", "resp", "modality", "env", "geom", "metal"]], on=["participant", "trial"])
cols = [c for c in F.columns if c not in ("participant", "trial", "resp", "modality", "n_frames_total", "env", "geom", "metal")]
conds = (F.env + "_" + F.geom + "_" + F.metal.astype(str)).values
F["log_dur"] = np.log(F.n_frames_total)
y = F.resp.values
trials = F.trial.values
pids = F.participant.values
by_trial = {t: np.where(trials == t)[0] for t in np.unique(trials)}

def within_trial(pred):
    ok = tot = 0.0
    for t, idx in by_trial.items():
        pos, neg = idx[y[idx] == 1], idx[y[idx] == 0]
        for i in pos:
            ok += np.sum(pred[i] > pred[neg]) + 0.5 * np.sum(pred[i] == pred[neg]); tot += len(neg)
    return ok / tot

Z = F.groupby("participant")[cols + ["log_dur"]].transform(lambda s: (s - s.mean()) / (s.std() + 1e-12))
rows = []
for c in cols + ["log_dur"]:
    a = within_trial(Z[c].values)
    rows.append(dict(feature=c, within_trial_auc=a, abs_dev=abs(a - 0.5)))
R = pd.DataFrame(rows).sort_values("abs_dev", ascending=False)
R.to_csv(RESULTS / "models" / "within_trial_features.csv", index=False)
print("single statistics, within-trial AUC (>0.5: higher value -> 'metal'):")
print(R.head(15).round(3).to_string(index=False))
print("\nlog duration alone:", round(float(R[R.feature == "log_dur"].within_trial_auc.iloc[0]), 3))

def lopo(X):
    """Leave-one-stimulus-out scores, as evaluate.loco: one model per stimulus, fitted to all
    videos of the other stimuli (the name is kept so that the rest of the script reads as before)."""
    pred = np.zeros(len(y))
    for c in np.unique(conds):
        te = conds == c
        m = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=5000)).fit(X[~te], y[~te])
        pred[te] = m.decision_function(X[te])
    return pred

X = Z[cols].values
d = Z["log_dur"].values.reshape(-1, 1)
Xr = X - LinearRegression().fit(d, X).predict(d)
ctrl = [dict(modality="all", full=within_trial(lopo(X)), no_duration=within_trial(lopo(Xr)), duration_only=within_trial(lopo(d)))]
print("full model within-trial AUC:", round(ctrl[0]["full"], 3))
print("duration partialled out    :", round(ctrl[0]["no_duration"], 3))
print("duration only (logistic)   :", round(ctrl[0]["duration_only"], 3))
for mod in ("hand", "head", "both"):
    sel = F.modality.values == mod
    sub = {t: i for t, i in by_trial.items() if sel[i[0]]}
    def wt(pred):
        ok = tot = 0.0
        for t, idx in sub.items():
            pos, neg = idx[y[idx] == 1], idx[y[idx] == 0]
            for i in pos:
                ok += np.sum(pred[i] > pred[neg]); tot += len(neg)
        return ok / tot
    ctrl.append(dict(modality=mod, full=wt(lopo(X)), no_duration=wt(lopo(Xr)), duration_only=wt(lopo(d))))
    print(f"  {mod}: full {ctrl[-1]['full']:.3f}  no-duration {ctrl[-1]['no_duration']:.3f}  duration-only {ctrl[-1]['duration_only']:.3f}")
pd.DataFrame(ctrl).to_csv(RESULTS / "models" / "duration_control.csv", index=False)

# per-observer values of the duration control, by condition
p_full, p_nd, p_d = lopo(X), lopo(Xr), lopo(d)
def wt_obs(pred, p, sel=None):
    ok = tot = 0.0
    for t, idx in by_trial.items():
        if sel is not None and not sel[idx[0]]: continue
        pos, neg = idx[y[idx] == 1], idx[y[idx] == 0]
        for i in pos:
            for j in neg:
                if pids[i] == p or pids[j] == p:
                    ok += float(pred[i] > pred[j]) + 0.5 * float(pred[i] == pred[j]); tot += 1
    return ok / tot if tot else np.nan
obs_rows = []
for p in PIDS:
    for mod in ("all", "hand", "head", "both"):
        sel = None if mod == "all" else (F.modality.values == mod)
        obs_rows.append(dict(participant=p, modality=mod, full=wt_obs(p_full, p, sel), no_duration=wt_obs(p_nd, p, sel), duration_only=wt_obs(p_d, p, sel)))
pd.DataFrame(obs_rows).to_csv(RESULTS / "models" / "duration_control_by_observer.csv", index=False)

# which kinds of statistics carry the information: the disagreement test on subsets of the statistics
is_area = [c.startswith("area_") for c in cols]
is_temporal = [c.endswith("_late_minus_early") or c.endswith("_ac1") or c.endswith("_sd") or c.endswith("_max") for c in cols]
is_mean = [c.endswith("_mean") for c in cols]
is_relational = [c.startswith("obj_bg_") or c.startswith("bg_") for c in cols]
is_flow = [c.split("_")[0] in ("coverage", "div", "sfm", "flow") for c in cols]
abl = []
for name, keep in [("full", [True] * len(cols)),
                   ("no_area", [not a for a in is_area]),
                   ("no_area_no_temporal", [not a and not t for a, t in zip(is_area, is_temporal)]),
                   ("means_only", is_mean),
                   ("no_relational", [not r and not f for r, f in zip(is_relational, is_flow)]),
                   ("relational_only", is_relational),
                   ("no_flow", [not f for f in is_flow]),
                   ("flow_only", is_flow)]:
    if sum(keep) == 0:
        continue
    abl.append(dict(model=name, n_features=int(sum(keep)), within_trial_auc=within_trial(lopo(X[:, keep]))))
    print(f"  {name:22s} {abl[-1]['n_features']:3d} features  within-trial AUC {abl[-1]['within_trial_auc']:.3f}")
pd.DataFrame(abl).to_csv(RESULTS / "models" / "feature_ablations.csv", index=False)
