"""The disagreement test as a whole and broken down by stimulus factor.

For the main model (the leave-one-stimulus-out scores of evaluate.py):
the AUC over all disagreeing pairs with its permutation null (responses
permuted among the observers of each trial, N_PERM times; every null value
is saved, for Figure 7b), the per-observer AUC, and the same AUC for the
pairs of each illumination environment and of each shape, each with its own
null and per-observer values.
Writes results/models/within_trial_breakdown.csv (one row per group),
       results/models/within_trial_null_all.csv (the null values, all pairs),
       results/models/within_trial_breakdown_by_observer.csv (observer x group).
"""
import numpy as np
import pandas as pd
from common import RESULTS, MAIN_FEATURES
from evaluate import T

N_PERM = 300
S = pd.read_csv(RESULTS / "models" / f"{MAIN_FEATURES}_C0.1_scores.csv", dtype={"participant": str})
S = S.merge(T[["participant", "trial", "env", "geom", "metal"]], on=["participant", "trial"])
trials = S.trial.values; y = S.resp.values; score = S.score_wt.values; pid = S.participant.values
by_trial = {t: np.where(trials == t)[0] for t in np.unique(trials)}
trial_env = S.groupby("trial").env.first(); trial_geom = S.groupby("trial").geom.first()
groups = {"all": list(by_trial)}
for e in ["Check", "Gray", "Sky", "Studio"]:
    groups[f"env_{e}"] = [t for t in by_trial if trial_env[t] == e]
for g in ["Bumpy", "Smooth"]:
    groups[f"geom_{g}"] = [t for t in by_trial if trial_geom[t] == g]


def pairs(yy, ts):
    ok, pi, pj = [], [], []
    for t in ts:
        idx = by_trial[t]
        pos, neg = idx[yy[idx] == 1], idx[yy[idx] == 0]
        for i in pos:
            for j in neg:
                ok.append(float(score[i] > score[j]) + 0.5 * float(score[i] == score[j])); pi.append(pid[i]); pj.append(pid[j])
    return np.array(ok), np.array(pi), np.array(pj)


rng = np.random.RandomState(0)
perms = []
for _ in range(N_PERM):
    yp = y.copy()
    for t, idx in by_trial.items():
        yp[idx] = rng.permutation(yp[idx])
    perms.append(yp)
rows, obs = [], {}
pids = sorted(set(pid))
for name, ts in groups.items():
    ok, pi, pj = pairs(y, ts)
    null = np.array([pairs(yp, ts)[0].mean() for yp in perms])
    if name == "all":
        pd.DataFrame({"null_auc": null}).to_csv(RESULTS / "models" / "within_trial_null_all.csv", index=False)
    per = pd.Series({p: ok[(pi == p) | (pj == p)].mean() for p in pids})
    obs[name] = per
    rows.append(dict(group=name, n_pairs=len(ok), within_trial_auc=ok.mean(), null_mean=null.mean(), null_sd=null.std(),
                     z=(ok.mean() - null.mean()) / null.std(), p=(np.sum(null >= ok.mean()) + 1) / (N_PERM + 1),
                     obs_mean=per.mean(), obs_se=per.std() / np.sqrt(len(per)), n_obs_above=int((per > 0.5).sum())))
R = pd.DataFrame(rows)
R.to_csv(RESULTS / "models" / "within_trial_breakdown.csv", index=False)
O = pd.DataFrame(obs); O.index.name = "participant"
O.to_csv(RESULTS / "models" / "within_trial_breakdown_by_observer.csv")
print(R.round(3).to_string(index=False))
