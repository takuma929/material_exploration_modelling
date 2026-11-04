"""Collect the model outputs into the two tables the figure scripts read.

results/summary/per_observer_auc.csv   one row per observer x predictor: the AUC of the
                                       predictor for that observer's responses, over all
                                       trials (auc_all) and within each exploration
                                       condition (auc_hand, auc_head, auc_both)
results/summary/model_summary.csv      one row per evaluated feature set, the first output
                                       of evaluate.py (results/models/<feature_set>_C<C>.csv)

The predictors are the two reference predictors (the physical metallicity of the
stimulus; the proportion of the other 18 observers who said "metal", and of all 19
including the observer, which bound the noise ceiling, Nili et al., 2014) and the
leave-one-observer-out scores of the models.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from common import ROOT, RESULTS, MAIN_FEATURES

OUT = RESULTS / "summary"; OUT.mkdir(exist_ok=True)
T = pd.read_csv(ROOT / "data" / "trials.csv", dtype={"participant": str})
T["others"] = T.groupby("trial").resp.transform(lambda s: (s.sum() - s) / (len(s) - 1))   # the other observers: lower bound of the ceiling
T["all_obs"] = T.groupby("trial").resp.transform("mean")                                  # the observer included: upper bound

# predictor name -> (column holding the prediction, scores file in results/models/ or None for a column of trials.csv)
PRED = {"metal_parameter": ("metal", None), "other_observers": ("others", None), "all_observers": ("all_obs", None),
        "image_stats": ("score", f"{MAIN_FEATURES}_C0.1_scores.csv"),                       # the main model (116 statistics, whole trial)
        "image_stats_first2s": ("score", f"{MAIN_FEATURES}_first2s_C0.1_scores.csv"),       # the same statistics from part of the trial
        "image_stats_early_half": ("score", f"{MAIN_FEATURES}_early_half_C0.1_scores.csv"),
        "image_stats_late_half": ("score", f"{MAIN_FEATURES}_late_half_C0.1_scores.csv"),
        "object_relational": ("score", "object_relational_C0.1_scores.csv"),                # the 100 image statistics without the motion statistics
        "image_stats_object": ("score", "image_stats_object_C0.1_scores.csv"),              # one family alone
        "relational": ("score", "relational_C0.1_scores.csv"),
        "flow": ("score", "flow_C0.1_scores.csv")}
rows = []
for name, (col, f) in PRED.items():
    if f is None:
        D = T[["participant", "trial", "resp", "modality", col]].rename(columns={col: "pred"})
    else:
        if not (RESULTS / "models" / f).exists():
            print("missing", f); continue
        S = pd.read_csv(RESULTS / "models" / f, dtype={"participant": str})
        D = S.merge(T[["participant", "trial", "modality"] + (["resp"] if "resp" not in S.columns else [])], on=["participant", "trial"]).rename(columns={col: "pred"})
    for p, g in D.groupby("participant"):
        r = dict(participant=p, predictor=name, auc_all=roc_auc_score(g.resp, g.pred))
        for mod, gm in g.groupby("modality"):
            r[f"auc_{mod}"] = roc_auc_score(gm.resp, gm.pred) if 0 < gm.resp.mean() < 1 else np.nan
        rows.append(r)
A = pd.DataFrame(rows)
A.to_csv(OUT / "per_observer_auc.csv", index=False)
print(A.groupby("predictor")[["auc_all", "auc_hand", "auc_head", "auc_both"]].mean().round(3))
M = pd.concat([pd.read_csv(f) for f in sorted((RESULTS / "models").glob("*_C*.csv")) if not (f.name.endswith("_scores.csv") or f.name.endswith("_observer_auc.csv"))], ignore_index=True)
M.to_csv(OUT / "model_summary.csv", index=False)
print(M[["feature_set", "n_features", "C", "auc_lopo", "within_trial_auc", "within_trial_z", "auc_lopo_swapped"]].round(3).to_string(index=False))
