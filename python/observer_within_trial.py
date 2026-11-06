"""The disagreement test per observer, and paired tests between models across observers.

For every model with saved scores, the AUC of the disagreement test restricted to
the pairs that contain a given observer (that observer against each observer who
saw the same trial and responded differently), so that observers become the unit
of the comparisons between models.  The scores are the leave-one-stimulus-out
scores of evaluate.py (score_wt), as in the disagreement test itself.

Writes results/models/within_trial_by_observer.csv   observer x model
       results/models/within_trial_tests.csv         one-sample tests against chance and
                                                     paired t-tests between models, plus the
                                                     paired tests of the leave-one-observer-out
                                                     AUCs (own video against the swapped video,
                                                     the reference predictors and the ceiling)
"""
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, ttest_1samp
from common import ROOT, RESULTS, MAIN_FEATURES

T = pd.read_csv(ROOT / "data" / "trials.csv", dtype={"participant": str})
M = RESULTS / "models"
# model name -> (scores file, column): score_wt = own video, score_wt_swapped = another observer's
# video of the same trial, score_plus = own video plus the other observers' responses
MODELS = {"stats": (f"{MAIN_FEATURES}_C0.1_scores.csv", "score_wt"),
          "stats_swapped": (f"{MAIN_FEATURES}_C0.1_scores.csv", "score_wt_swapped"),
          "stats_plus_others": (f"{MAIN_FEATURES}_C0.1_scores.csv", "score_plus"),
          "stats_noflow": ("object_relational_C0.1_scores.csv", "score_wt"),          # the 100 image statistics without the motion statistics
          "object_only": ("image_stats_object_C0.1_scores.csv", "score_wt"),          # one family alone
          "relational_only": ("relational_C0.1_scores.csv", "score_wt"),
          "flow_only": ("flow_C0.1_scores.csv", "score_wt")}
# paired comparisons between models (a minus b)
CONTRASTS = [("stats", "stats_swapped"), ("stats", "stats_noflow"), ("stats", "object_only"), ("stats", "relational_only"),
             ("stats", "flow_only")]


def per_observer_within_trial(S, col):
    """For each observer, the proportion of their disagreeing pairs in which the video of
    the observer who said "metal" scored higher (ties count one half)."""
    S = S.merge(T[["participant", "trial", "resp", "modality"]], on=["participant", "trial"], suffixes=("", "_t"))
    if "resp" not in S.columns:
        S["resp"] = S["resp_t"]
    out = {}
    for p in sorted(S.participant.unique()):
        ok = tot = 0.0
        for t, g in S.groupby("trial"):
            me = g[g.participant == p]
            if len(me) != 1:
                continue
            others = g[(g.participant != p) & (g.resp != me.resp.iloc[0])]
            if len(others) == 0:
                continue
            s_me = me[col].iloc[0]; s_ot = others[col].values
            if me.resp.iloc[0] == 1:
                ok += np.sum(s_me > s_ot) + 0.5 * np.sum(s_me == s_ot)
            else:
                ok += np.sum(s_me < s_ot) + 0.5 * np.sum(s_me == s_ot)
            tot += len(others)
        out[p] = ok / tot
    return pd.Series(out)


rows = {}
for name, (f, col) in MODELS.items():
    if not (M / f).exists():
        print("missing", f); continue
    S = pd.read_csv(M / f, dtype={"participant": str})
    if col not in S.columns:
        print("no column", col, "in", f); continue
    rows[name] = per_observer_within_trial(S, col)
W = pd.DataFrame(rows); W.index.name = "participant"
W.to_csv(M / "within_trial_by_observer.csv")
print(W.mean().round(3).to_string())


def record(contrast, a, b_mean, b=None):
    """One row of the test table: a against a constant (one-sample t) or against b (paired t)."""
    t = ttest_1samp(a, b_mean) if b is None else ttest_rel(a, b)
    n_pos = int((a > b_mean).sum()) if b is None else int((np.asarray(a) > np.asarray(b)).sum())
    tests.append(dict(contrast=contrast, mean_a=float(np.mean(a)), mean_b=float(b_mean if b is None else np.mean(b)),
                      se_a=float(np.std(a, ddof=1) / np.sqrt(len(a))), t=float(t.statistic), df=len(a) - 1, p=float(t.pvalue), n_positive=n_pos))


tests = []
for name in W.columns:
    record(f"{name} vs 0.5", W[name], 0.5)
for a, b in CONTRASTS:
    if a in W.columns and b in W.columns:
        record(f"{a} vs {b}", W[a], None, W[b])
# leave-one-observer-out AUCs of the main model (evaluate.py's third output): the own video against the
# swapped video, against the physical metallicity and against the other observers (the lower bound of
# the ceiling); and own video plus the other observers' responses against the other observers alone
f = M / f"{MAIN_FEATURES}_C0.1_observer_auc.csv"
if f.exists():
    O = pd.read_csv(f, dtype={"participant": str})
    for a, b in [("auc_own", "auc_swapped"), ("auc_plus_condition", "auc_others"), ("auc_own", "auc_others"), ("auc_own", "auc_metal")]:
        record(f"{a} vs {b}", O[a], None, O[b])
Tt = pd.DataFrame(tests)
Tt.to_csv(M / "within_trial_tests.csv", index=False)
print(Tt.round(4).to_string(index=False))
