"""Does the within-trial test succeed when the two videos differ?

Among pairs of observers who saw the same trial but responded differently,
the within-trial AUC is the proportion of pairs in which the own-video model
scored the video of the observer who said "metal" higher.  Here the pairs
are split by how much the two videos differ in model score (quartiles of the
absolute difference), with a permutation null (responses permuted within
trials, quartile boundaries fixed) for each quartile, and also by physical
metallicity.  A swapped-video control replaces each video of a pair by another
observer's video of the same trial.
Writes results/models/within_trial_by_difference.csv              one row per quarter
       results/models/within_trial_by_difference_by_observer.csv  observer x quarter
       results/models/within_trial_by_level.csv                   by physical metallicity
"""
import itertools
import numpy as np
import pandas as pd
from common import ROOT, RESULTS, MAIN_FEATURES

N_PERM = 300
N_Q = 4
S = pd.read_csv(RESULTS / "models" / f"{MAIN_FEATURES}_C0.1_scores.csv", dtype={"participant": str})
T = pd.read_csv(ROOT / "data" / "trials.csv", dtype={"participant": str})
S = S.merge(T[["participant", "trial", "metal"]], on=["participant", "trial"])
trials = S.trial.values; y = S.resp.values; score = S.score_wt.values; metal = S.metal.values   # leave-one-stimulus-out scores
by_trial = {t: np.where(trials == t)[0] for t in np.unique(trials)}

def pairs(yy):
    """(|score difference|, metal responder scored higher, metallic level) over disagreeing pairs."""
    d, ok, lv = [], [], []
    for t, idx in by_trial.items():
        pos, neg = idx[yy[idx] == 1], idx[yy[idx] == 0]
        for i in pos:
            for j in neg:
                d.append(abs(score[i] - score[j])); ok.append(float(score[i] > score[j]) + 0.5 * float(score[i] == score[j])); lv.append(metal[i])
    return np.array(d), np.array(ok), np.array(lv)

d, ok, lv = pairs(y)
edges = np.quantile(d, np.linspace(0, 1, N_Q + 1)); edges[0], edges[-1] = 0, np.inf
q = np.clip(np.searchsorted(edges, d, side="right") - 1, 0, N_Q - 1)
rng = np.random.RandomState(0)
null = np.full((N_PERM, N_Q), np.nan)
for k in range(N_PERM):
    yp = y.copy()
    for t, idx in by_trial.items():
        yp[idx] = rng.permutation(yp[idx])
    dp, okp, _ = pairs(yp)
    qp = np.clip(np.searchsorted(edges, dp, side="right") - 1, 0, N_Q - 1)
    null[k] = [okp[qp == i].mean() for i in range(N_Q)]
rows = []
for i in range(N_Q):
    sel = q == i
    rows.append(dict(quartile=i + 1, lo=edges[i], hi=edges[i + 1], d_median=np.median(d[sel]), n=int(sel.sum()),
                     within_trial_auc=ok[sel].mean(), null_mean=null[:, i].mean(), null_sd=null[:, i].std()))
    rows[-1]["z"] = (rows[-1]["within_trial_auc"] - rows[-1]["null_mean"]) / rows[-1]["null_sd"]
    rows[-1]["p"] = (np.sum(null[:, i] >= rows[-1]["within_trial_auc"]) + 1) / (N_PERM + 1)
R = pd.DataFrame(rows)

# Swapped videos: each member of a pair is replaced by a random other observer's
# video of the same trial (its own leave-one-stimulus-out score), ten assignments.
# The pairs are split (i) by the difference of the OWN videos (the quartiles above)
# and (ii) by the difference of the swapped videos themselves, same boundaries.
N_SWAP = 10
pair_ij = []
for t, idx in by_trial.items():
    pos, neg = idx[y[idx] == 1], idx[y[idx] == 0]
    for i in pos:
        for j in neg:
            pair_ij.append((i, j))
pair_ij = np.array(pair_ij)
rs = np.random.RandomState(1)
sw_all = np.zeros(N_SWAP); sw_own = np.zeros((N_SWAP, N_Q)); sw_sw = np.zeros((N_SWAP, N_Q)); n_sw = np.zeros((N_SWAP, N_Q))
for k in range(N_SWAP):
    ds, oks = np.empty(len(pair_ij)), np.empty(len(pair_ij))
    for m, (i, j) in enumerate(pair_ij):
        idx = by_trial[trials[i]]
        a = rs.choice(idx[idx != i]); b = rs.choice(idx[(idx != j) & (idx != a)])
        ds[m] = abs(score[a] - score[b]); oks[m] = float(score[a] > score[b]) + 0.5 * float(score[a] == score[b])
    sw_all[k] = oks.mean()
    qs = np.clip(np.searchsorted(edges, ds, side="right") - 1, 0, N_Q - 1)
    sw_own[k] = [oks[q == n].mean() for n in range(N_Q)]
    sw_sw[k] = [oks[qs == n].mean() for n in range(N_Q)]
    n_sw[k] = [(qs == n).sum() for n in range(N_Q)]
R["swapped_auc_all"] = sw_all.mean()
R["swapped_auc_own_split"] = sw_own.mean(0); R["swapped_auc_own_split_sd"] = sw_own.std(0)
R["swapped_auc_swapped_split"] = sw_sw.mean(0); R["swapped_auc_swapped_split_sd"] = sw_sw.std(0); R["n_swapped_split"] = n_sw.mean(0)
R.to_csv(RESULTS / "models" / "within_trial_by_difference.csv", index=False)

# per observer, over the pairs that contain the observer
pid_i, pid_j = [], []
for t, idx in by_trial.items():
    pos, neg = idx[y[idx] == 1], idx[y[idx] == 0]
    for i in pos:
        for j in neg:
            pid_i.append(S.participant.values[i]); pid_j.append(S.participant.values[j])
pid_i, pid_j = np.array(pid_i), np.array(pid_j)
obs_rows = []
for p in sorted(set(S.participant)):
    has = (pid_i == p) | (pid_j == p)
    obs_rows.append(dict(participant=p, **{f"q{k + 1}": ok[has & (q == k)].mean() for k in range(N_Q)}))
pd.DataFrame(obs_rows).to_csv(RESULTS / "models" / "within_trial_by_difference_by_observer.csv", index=False)
print(f"{len(d)} disagreeing pairs, within-trial AUC {ok.mean():.3f} (swapped videos {sw_all.mean():.3f}); by quartile of |score difference|:")
print(R.round(3).to_string(index=False))
L = pd.DataFrame([dict(metal=m, n=int((lv == m).sum()), within_trial_auc=ok[lv == m].mean()) for m in np.unique(lv)])
L.to_csv(RESULTS / "models" / "within_trial_by_level.csv", index=False)
print("by metallic level:"); print(L.round(3).to_string(index=False))
