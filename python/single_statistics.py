"""Which image statistics predict the responses, for the group and for each observer.

For each of the whole-trial statistics, the AUC of the statistic alone
(z-scored within each observer) against the observer's responses over their
120 trials: a value above 0.5 means that a higher value went with a "metal"
response.  The group value is the mean over the 19 observers, with a
permutation null (responses shuffled within each observer, 300 shuffles).
Writes results/models/single_statistics.csv (group) and
results/models/single_statistics_by_observer.csv (participant x statistic).
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from common import RESULTS, MAIN_FEATURES
from evaluate import load_features_clean, T, PIDS

N_PERM = 300
F = load_features_clean(MAIN_FEATURES).merge(T[["participant", "trial", "resp"]], on=["participant", "trial"])
cols = [c for c in F.columns if c not in ("participant", "trial", "resp")]
Z = F.groupby("participant")[cols].transform(lambda s: (s - s.mean()) / (s.std() + 1e-12))
y = F.resp.values; pids = F.participant.values
obs = {p: pids == p for p in PIDS}

def auc_by_observer(x, yy):
    return np.array([roc_auc_score(yy[m], x[m]) for m in obs.values()])

A = np.array([auc_by_observer(Z[c].values, y) for c in cols])          # statistics x observers
rng = np.random.RandomState(0)
null = np.zeros((N_PERM, len(cols)))
for k in range(N_PERM):
    yp = y.copy()
    for m in obs.values():
        yp[m] = rng.permutation(yp[m])
    null[k] = [auc_by_observer(Z[c].values, yp).mean() for c in cols]
R = pd.DataFrame(dict(feature=cols, auc=A.mean(1), auc_sd=A.std(1), n_same_direction=[(np.sign(a - 0.5) == np.sign(g - 0.5)).sum() for a, g in zip(A, A.mean(1))],
                      null_mean=null.mean(0), null_sd=null.std(0)))
R["abs_dev"] = (R.auc - 0.5).abs()
R = R.sort_values("abs_dev", ascending=False)
R.to_csv(RESULTS / "models" / "single_statistics.csv", index=False)
B = pd.DataFrame(A.T, columns=cols); B.insert(0, "participant", list(PIDS))
B.to_csv(RESULTS / "models" / "single_statistics_by_observer.csv", index=False)
print("single statistics, AUC against the responses (mean over observers; > 0.5: higher value -> 'metal'):")
print(R.head(16).round(3).to_string(index=False))
best = B.set_index("participant")[cols]
sign = np.sign(R.set_index("feature").auc - 0.5)
aligned = 0.5 + (best - 0.5) * sign[cols].values
print("\nbest statistic per observer (sign aligned to the group):")
for p in PIDS:
    print(f"  {p}: {aligned.loc[p].idxmax():32s} {aligned.loc[p].max():.3f}")
print("distinct best:", aligned.idxmax(1).nunique(), "| best AUC range", round(aligned.max(1).min(), 3), "-", round(aligned.max(1).max(), 3))
