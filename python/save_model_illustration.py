"""Numbers for the model-illustration figure (fig2_model.m).

Refits the main model for the observer of the example video of Figure 2a
(OBS, TRIAL below; leave-one-observer-out, the same pipeline as evaluate.py)
and writes data/example_frames/model_illustration.csv: for every statistic
its value in that video (z-scored within the observer, then standardized over
the training videos, as in evaluate.py) and the fitted weight; extra rows
give the intercept, the observer, the trial, the video's score, the
observer's AUC and the within-trial counts of that trial.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from common import ROOT, MAIN_FEATURES
from evaluate import load_features_clean, T

OBS, TRIAL = "005", 123      # the video of Figure 2a (Sky, bumpy, physical metallicity 0.5, head and hand)
F = load_features_clean(MAIN_FEATURES).merge(T[["participant", "trial", "resp"]], on=["participant", "trial"])
cols = [c for c in F.columns if c not in ("participant", "trial", "resp")]
F[cols] = F.groupby("participant")[cols].transform(lambda s: (s - s.mean()) / (s.std() + 1e-12))   # as in evaluate.py
tr = F.participant != OBS
m = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=5000)).fit(F.loc[tr, cols].values, F.loc[tr, "resp"].values)
sc, lr = m.named_steps["standardscaler"], m.named_steps["logisticregression"]
def row(pid):
    x = F[(F.participant == pid) & (F.trial == TRIAL)][cols].values
    return sc.transform(x)[0], float(m.decision_function(x)[0])
xv, sv = row(OBS)
S = pd.read_csv(ROOT / "results" / "models" / f"{MAIN_FEATURES}_C0.1_scores.csv", dtype={"participant": str})
stored = float(S[(S.participant == OBS) & (S.trial == TRIAL)].score.iloc[0])
print(f"score of observer {OBS}, trial {TRIAL}: {sv:+.3f} (stored {stored:+.3f})")
st = S[S.trial == TRIAL]; pos, neg = st[st.resp == 1].score_wt.values, st[st.resp == 0].score_wt.values   # leave-one-stimulus-out, as in the disagreement test
ok = int(sum((a > b) for a in pos for b in neg))
A = pd.read_csv(ROOT / "results" / "models" / f"{MAIN_FEATURES}_C0.1_observer_auc.csv", dtype={"participant": str})
auc = float(A[A.participant == OBS].auc_own.iloc[0])
out = pd.DataFrame(dict(feature=cols, x=xv, w=lr.coef_[0]))
for k, v in [("intercept", float(lr.intercept_[0])), ("observer", int(OBS)), ("trial", TRIAL), ("score", sv), ("auc", auc),
             ("n_metal", len(pos)), ("n_glass", len(neg)), ("n_pairs", len(pos) * len(neg)), ("metal_higher", ok)]:
    out.loc[len(out)] = [k, np.nan, v]
out.to_csv(ROOT / "data" / "example_frames" / "model_illustration.csv", index=False)
print(out.tail(9).to_string(index=False)); print("n features", len(cols))
