"""How many dimensions of the 116 statistics the own-video model needs (Figure 4a).

Leave-one-observer-out logistic regression (C 0.1) as in evaluate.py, with a
PCA fitted inside each fold on the 18 training observers' videos and the
first k components kept. Writes results/models/pca_sweep.csv (mean AUC per k)
and results/models/pca_sweep_by_observer.csv (observer x k).
"""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from common import RESULTS, MAIN_FEATURES
from evaluate import load_features_clean, T, PIDS

KS = [1, 2, 3, 5, 8, 10, 15, 20, 30, 50, 100]
C = 0.1
F = load_features_clean(MAIN_FEATURES).merge(T[["participant", "trial", "resp"]], on=["participant", "trial"])   # the same cleaning as evaluate.py
good = [c for c in F.columns if c not in ("participant", "trial", "resp")]
Z = F.groupby("participant")[good].transform(lambda s: (s - s.mean()) / (s.std() + 1e-12)).values
y = F.resp.values; pids = F.participant.values
rows = []
for k in KS:
    for p in PIDS:
        te = pids == p
        if te.sum() == 0:
            continue
        steps = [StandardScaler()] + ([PCA(k, random_state=0)] if k < Z.shape[1] else []) + [LogisticRegression(C=C, max_iter=5000)]
        m = make_pipeline(*steps).fit(Z[~te], y[~te])
        rows.append(dict(k=k, participant=p, auc=roc_auc_score(y[te], m.decision_function(Z[te]))))
    print(f"k = {k:3d}: LOPO AUC {np.mean([r['auc'] for r in rows if r['k'] == k]):.3f}")
D = pd.DataFrame(rows)
D.pivot(index="participant", columns="k", values="auc").add_prefix("k").reset_index().to_csv(RESULTS / "models" / "pca_sweep_by_observer.csv", index=False)
S = D.groupby("k").auc.agg(["mean", "std", "median"]).reset_index().rename(columns={"mean": "auc_mean", "std": "auc_sd", "median": "auc_median"})
# variance explained by the first k components (PCA on all videos, for the text)
pca = PCA().fit(StandardScaler().fit_transform(Z)); cum = np.cumsum(pca.explained_variance_ratio_)
S["var_explained"] = [cum[min(k, len(cum)) - 1] for k in S.k]
S.to_csv(RESULTS / "models" / "pca_sweep.csv", index=False)
print("variance explained:", ", ".join(f"k{k} {cum[k - 1]:.2f}" for k in KS if k <= len(cum)))
