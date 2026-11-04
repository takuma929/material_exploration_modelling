"""Behavioural summaries of the three exploration conditions (data/trials.csv).

Writes results/behaviour/
  trial_means.csv      per trial: the proportion of observers who said "metal"
  observer_summary.csv per observer: AUC of the other observers' responses and of the
                       physical metallicity for their responses (overall and by condition),
                       logistic slope against physical metallicity, P(metal), mean duration
  condition_means.csv  P(metal) by physical metallicity x condition / environment / shape
and prints the split-half reliability of the responses and the mean AUCs.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from common import ROOT, RESULTS

OUT = RESULTS / "behaviour"
OUT.mkdir(parents=True, exist_ok=True)
T = pd.read_csv(ROOT / "data" / "trials.csv", dtype={"participant": str})
T["dur_s"] = (T.t_last_ms - T.t_first_ms) / 1000
obs = sorted(T.participant.unique())
R = T.pivot(index="trial", columns="participant", values="resp")  # 120 x 19
cond = T.drop_duplicates("trial").set_index("trial")[["env", "geom", "metal", "modality"]]

# --- per-trial means and reliability
tm = cond.copy()
tm["p_metal"] = R.mean(axis=1)
tm["n_obs"] = R.notna().sum(axis=1)
tm.to_csv(OUT / "trial_means.csv")
rng = np.random.default_rng(0)
sh = []
for _ in range(500):
    perm = rng.permutation(len(obs))
    a, b = R.iloc[:, perm[:len(obs) // 2]].mean(axis=1), R.iloc[:, perm[len(obs) // 2:]].mean(axis=1)
    sh.append(np.corrcoef(a, b)[0, 1])
r_half = float(np.mean(sh))
r_sb = 2 * r_half / (1 + r_half)
print(f"split-half r of per-trial P(metal): {r_half:.3f}  (Spearman-Brown full-sample {r_sb:.3f})")
print(f"r(P(metal), metal level) = {np.corrcoef(tm.p_metal, tm.metal)[0,1]:.3f}")

# --- per observer: leave-one-out ceiling, accuracy, slopes
rows = []
for o in obs:
    y = R[o].values
    others = R.drop(columns=o).mean(axis=1).values
    auc_loo = roc_auc_score(y, others)
    auc_metal = roc_auc_score(y, R.index.map(cond.metal).values)
    row = dict(participant=o, auc_other_observers=auc_loo, auc_metal_level=auc_metal,
               p_metal=y.mean(), dur_s=T[T.participant == o].dur_s.mean())
    for mod in ("hand", "head", "both"):
        idx = cond.modality.values == mod
        x = cond.metal.values[idx].reshape(-1, 1)
        lr = LogisticRegression(C=1e6, max_iter=1000).fit(x, y[idx])
        row[f"slope_{mod}"] = lr.coef_[0, 0]           # logit units per unit metal
        row[f"auc_metal_{mod}"] = roc_auc_score(y[idx], x.ravel())
        row[f"auc_others_{mod}"] = roc_auc_score(y[idx], others[idx])
    rows.append(row)
S = pd.DataFrame(rows)
S.to_csv(OUT / "observer_summary.csv", index=False)
print("\nper-observer AUC, mean (sd):")
for c in ["auc_other_observers", "auc_metal_level", "auc_metal_hand", "auc_metal_head", "auc_metal_both",
          "auc_others_hand", "auc_others_head", "auc_others_both"]:
    print(f"  {c:22s} {S[c].mean():.3f} ({S[c].std():.3f})")

# --- condition means
cm = []
for fac in ("modality", "env", "geom"):
    g = T.groupby([fac, "metal"]).resp.mean().unstack("metal")
    g.insert(0, "factor", fac)
    cm.append(g.reset_index().rename(columns={fac: "level"}))
CM = pd.concat(cm, ignore_index=True)
CM.to_csv(OUT / "condition_means.csv", index=False)
print("\nP(metal) by metal level:")
print(CM.round(2).to_string(index=False))
print("\nexploration duration (s) by modality:", T.groupby("modality").dur_s.mean().round(2).to_dict())
print("exploration duration (s) by metal:", T.groupby("metal").dur_s.mean().round(2).to_dict())
