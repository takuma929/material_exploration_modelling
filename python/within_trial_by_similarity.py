"""Does the disagreement test succeed when the two videos differ in their
statistics?  As within_trial_by_difference.py, but the pairs are split by
how far apart the two videos are in the space of the statistics: the 116
standardized statistics reduced by PCA to the components that carry 95% of
their variance, Euclidean distance between the two videos' component scores.
Quartiles of that distance, each with its permutation null (responses
permuted within trials, boundaries fixed), and per observer.
Writes results/models/within_trial_by_similarity.csv              one row per quarter (Figure 7c)
       results/models/within_trial_by_similarity_by_observer.csv  observer x quarter
       results/models/within_trial_pairs.csv                      every disagreeing pair: distance, score difference
       results/models/within_trial_by_similarity_deciles.csv      the same split in ten
"""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from common import ROOT, RESULTS, MAIN_FEATURES
from evaluate import load_features_clean, T

N_PERM = 300
N_Q = 4
VAR = 0.95
F = load_features_clean(MAIN_FEATURES)                       # the same cleaning as evaluate.py
good = [c for c in F.columns if c not in ("participant", "trial")]
Z = F.groupby("participant")[good].transform(lambda s: (s - s.mean()) / (s.std() + 1e-12)).values   # z-scored within observer, as in evaluate.py
pca = PCA().fit(StandardScaler().fit_transform(Z))
k = int(np.searchsorted(np.cumsum(pca.explained_variance_ratio_), VAR) + 1)
P = PCA(k).fit_transform(StandardScaler().fit_transform(Z))
print(f"{k} components carry {np.cumsum(pca.explained_variance_ratio_)[k - 1]:.3f} of the variance of the {Z.shape[1]} statistics")
S = pd.read_csv(RESULTS / "models" / f"{MAIN_FEATURES}_C0.1_scores.csv", dtype={"participant": str})
S = F[["participant", "trial"]].merge(S, on=["participant", "trial"]).merge(T[["participant", "trial", "metal"]], on=["participant", "trial"])
assert len(S) == len(F)
trials = S.trial.values; y = S.resp.values; score = S.score_wt.values
by_trial = {t: np.where(trials == t)[0] for t in np.unique(trials)}

def pairs(yy):
    d, ok, ds = [], [], []
    for t, idx in by_trial.items():
        pos, neg = idx[yy[idx] == 1], idx[yy[idx] == 0]
        for i in pos:
            for j in neg:
                d.append(np.linalg.norm(P[i] - P[j])); ds.append(abs(score[i] - score[j]))
                ok.append(float(score[i] > score[j]) + 0.5 * float(score[i] == score[j]))
    return np.array(d), np.array(ok), np.array(ds)

d, ok, ds = pairs(y)
print(f"{len(d)} disagreeing pairs; correlation between the distance in statistics and |score difference|: r = {np.corrcoef(d, ds)[0, 1]:.3f} (Spearman {pd.Series(d).corr(pd.Series(ds), method='spearman'):.3f})")
edges = np.quantile(d, np.linspace(0, 1, N_Q + 1)); edges[0], edges[-1] = 0, np.inf
q = np.clip(np.searchsorted(edges, d, side="right") - 1, 0, N_Q - 1)
rng = np.random.RandomState(0)
null = np.full((N_PERM, N_Q), np.nan)
for n_ in range(N_PERM):
    yp = y.copy()
    for t, idx in by_trial.items():
        yp[idx] = rng.permutation(yp[idx])
    dp, okp, _ = pairs(yp)
    qp = np.clip(np.searchsorted(edges, dp, side="right") - 1, 0, N_Q - 1)
    null[n_] = [okp[qp == i].mean() for i in range(N_Q)]
rows = []
for i in range(N_Q):
    sel = q == i
    r = dict(quartile=i + 1, lo=edges[i], hi=edges[i + 1], d_median=np.median(d[sel]), n=int(sel.sum()),
             within_trial_auc=ok[sel].mean(), null_mean=null[:, i].mean(), null_sd=null[:, i].std(), score_diff_median=np.median(ds[sel]))
    r["z"] = (r["within_trial_auc"] - r["null_mean"]) / r["null_sd"]; r["p"] = (np.sum(null[:, i] >= r["within_trial_auc"]) + 1) / (N_PERM + 1)
    rows.append(r)
R = pd.DataFrame(rows); R["n_components"] = k; R["var_explained"] = float(np.cumsum(pca.explained_variance_ratio_)[k - 1])
R.to_csv(RESULTS / "models" / "within_trial_by_similarity.csv", index=False)
print(R.round(3).to_string(index=False))
# per observer
pid_i, pid_j = [], []
for t, idx in by_trial.items():
    pos, neg = idx[y[idx] == 1], idx[y[idx] == 0]
    for i in pos:
        for j in neg:
            pid_i.append(S.participant.values[i]); pid_j.append(S.participant.values[j])
pid_i, pid_j = np.array(pid_i), np.array(pid_j)
obs = []
for p in sorted(set(S.participant)):
    has = (pid_i == p) | (pid_j == p)
    obs.append(dict(participant=p, **{f"q{i + 1}": ok[has & (q == i)].mean() for i in range(N_Q)}))
pd.DataFrame(obs).to_csv(RESULTS / "models" / "within_trial_by_similarity_by_observer.csv", index=False)
# every disagreeing pair, for Figure 7b: distance, signed score difference (metal responder minus glass responder)
rows_p = []
for t, idx in by_trial.items():
    pos, neg = idx[y[idx] == 1], idx[y[idx] == 0]
    for i in pos:
        for j in neg:
            rows_p.append(dict(trial=t, metal_obs=S.participant.values[i], glass_obs=S.participant.values[j],
                               distance=np.linalg.norm(P[i] - P[j]), score_diff=score[i] - score[j]))
Pp = pd.DataFrame(rows_p); Pp["quartile"] = np.clip(np.searchsorted(edges, Pp.distance.values, side="right") - 1, 0, N_Q - 1) + 1
Pp.to_csv(RESULTS / "models" / "within_trial_pairs.csv", index=False)
dec_edges = np.quantile(d, np.linspace(0, 1, 11)); dec_edges[0], dec_edges[-1] = 0, np.inf
dec = np.clip(np.searchsorted(dec_edges, d, side="right") - 1, 0, 9)
pd.DataFrame([dict(decile=i + 1, lo=dec_edges[i], hi=dec_edges[i + 1], d_median=np.median(d[dec == i]), n=int((dec == i).sum()),
                   within_trial_auc=ok[dec == i].mean()) for i in range(10)]).to_csv(RESULTS / "models" / "within_trial_by_similarity_deciles.csv", index=False)
print("score difference percentiles 1/5/50/95/99:", np.round(np.percentile(Pp.score_diff, [1, 5, 50, 95, 99]), 2), " distance 1/99:", np.round(np.percentile(d, [1, 99]), 2))
print("by decile of distance:", np.round([ok[dec == i].mean() for i in range(10)], 3))
