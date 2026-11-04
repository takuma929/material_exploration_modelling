"""Common evaluation of a feature set as a predictor of the observers' metal
responses (Experiment 3).

Protocol (all L2 logistic regression, C = 0.1, every statistic z-scored within observer,
then standardized over the training videos; sklearn):
  LOPO          leave-one-observer-out: fitted to 18 observers, scores the 19th; AUC per
                held-out observer against that observer's responses, then the mean
  within        5-fold cross-validation inside each observer (for reference)
  ceiling       the proportion of the other 18 observers who said "metal" on the same trial
                (lower bound of the noise ceiling); collect_results.py adds the upper bound
  metal         the physical metallicity (rendering parameter) alone
  within-trial  the disagreement test: pairwise AUC among observers who saw
                the same trial ID but responded differently, with a
                permutation null (responses shuffled among observers within
                trial).  Scored leave-one-stimulus-out: one model per
                stimulus (environment x shape x physical metallicity), fitted
                to the videos and responses of all observers on the other
                stimuli in every block, so that no response to the stimulus
                being scored enters the model and a model that read only the
                stimulus would score the videos of a trial identically
                (leave-one-observer-out scores carry the held-out observer's
                own bias and the other observers' responses to the same
                trial, which pushes the within-trial AUC below 0.5 even for
                identical videos: 0.35 in the static block)
  swapped       LOPO with the features of another observer's video of the
                same trial ID
  by modality   LOPO AUC and within-trial AUC per exploration block

Usage: python evaluate.py <feature_set> [--C 0.1]
  feature_set: object_relational_flow[_first2s|_early_half|_late_half]   the main model: 116 statistics
               object_relational[...]                                   the 100 image statistics without motion
               image_stats_object[...] | relational[...] | flow[...]    one family alone
  (a suffix selects the per-trial summaries of a time window, results/features/<family>_<window>/)
Writes results/models/<feature_set>_C<C>.csv               one row: every AUC above
       results/models/<feature_set>_C<C>_scores.csv        per video: the leave-one-observer-out score
                                                           (score), the same with the other observers'
                                                           responses added (score_plus), of another
                                                           observer's video (score_swapped), and the
                                                           leave-one-stimulus-out scores (score_wt, *_swapped)
       results/models/<feature_set>_C<C>_observer_auc.csv  per observer: AUC of the own video, of the own
                                                           video plus the other observers, of the other
                                                           observers, of the physical metallicity, of the swapped video
"""
import argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from common import ROOT, RESULTS, FEATURES

T = pd.read_csv(ROOT / "data" / "trials.csv", dtype={"participant": str})
PIDS = sorted(T.participant.unique())


def load_features(name):
    """The per-trial statistics of a feature set: a DataFrame with participant, trial and one
    column per statistic.  Combined sets are the merge of their families, so that
    object_relational_flow<suffix> = image_stats_object<suffix> + relational<suffix> + flow<suffix>."""
    if name.startswith("object_relational_flow"):
        suffix = name[len("object_relational_flow"):]
        return load_features("object_relational" + suffix).merge(load_features("flow" + suffix), on=["participant", "trial"])
    if name.startswith("object_relational"):
        suffix = name[len("object_relational"):]
        A = load_features("image_stats_object" + suffix); B = load_features("relational" + suffix)
        return A.merge(B, on=["participant", "trial"])
    if name.startswith(("image_stats", "relational", "flow")):
        parts = [pd.read_csv(f, dtype={"participant": str}) for f in sorted((FEATURES / name).glob("participant_*.csv"))]
        F = pd.concat(parts, ignore_index=True)
        cols = [c for c in F.columns if c not in ("participant", "trial", "n_frames_used", "n_frames_total")]
        return F[["participant", "trial"] + cols]
    raise ValueError(name)


def load_features_clean(name):
    """load_features with the cleaning of evaluate(): inf -> NaN, columns with more than 1% missing
    or no variance dropped (e.g. the flow summaries undefined for the shortest trials), rows with
    a missing value dropped."""
    F = load_features(name)
    cols = [c for c in F.columns if c not in ("participant", "trial")]
    F[cols] = F[cols].replace([np.inf, -np.inf], np.nan)
    good = [c for c in cols if F[c].notna().mean() > 0.99 and F[c].std() > 0]
    return F[["participant", "trial"] + good].dropna(subset=good).reset_index(drop=True)


def evaluate(name, C=0.1, n_perm=300, n_swap=10, pca=None, seed=0):
    F = load_features(name)
    cols = [c for c in F.columns if c not in ("participant", "trial")]
    F[cols] = F[cols].replace([np.inf, -np.inf], np.nan)
    good = [c for c in cols if F[c].notna().mean() > 0.99 and F[c].std() > 0]
    F = F.dropna(subset=good).merge(T[["participant", "trial", "resp", "metal", "modality", "env", "geom"]],
                                    on=["participant", "trial"]).reset_index(drop=True)
    Z = F.groupby("participant")[good].transform(lambda s: (s - s.mean()) / (s.std() + 1e-12)).values
    if pca:
        Z = PCA(pca, random_state=seed).fit_transform(Z)
    y = F.resp.values.copy()
    pids = F.participant.values
    trials = F.trial.values
    others = F.groupby("trial").resp.transform(lambda s: (s.sum() - s) / (len(s) - 1)).values
    by_trial = {t: np.where(trials == t)[0] for t in np.unique(trials)}
    conds = (F.env + "_" + F.geom + "_" + F.metal.astype(str)).values     # the 40 stimuli
    rng = np.random.default_rng(seed)

    def lopo(X):
        pred = np.zeros(len(y))
        for p in PIDS:
            te = pids == p
            if te.sum() == 0: continue
            m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000)).fit(X[~te], y[~te])
            pred[te] = m.decision_function(X[te])
        return pred

    def loco(X):
        """Leave-one-stimulus-out scores for the disagreement test (see the module docstring)."""
        pred = np.zeros(len(y))
        for c in np.unique(conds):
            te = conds == c
            m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000)).fit(X[~te], y[~te])
            pred[te] = m.decision_function(X[te])
        return pred

    def within_cv(X):
        aucs = []
        for p in PIDS:
            idx = np.where(pids == p)[0]
            if len(idx) == 0: continue
            pr = np.zeros(len(idx))
            for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(idx, y[idx]):
                m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000)).fit(X[idx[tr]], y[idx[tr]])
                pr[te] = m.decision_function(X[idx[te]])
            aucs.append(roc_auc_score(y[idx], pr))
        return float(np.mean(aucs))

    def auc_obs(pred, sel=None, per_observer=False):
        out = {}
        for p in PIDS:
            m = (pids == p) if sel is None else (pids == p) & sel
            if m.sum() and 0 < y[m].mean() < 1:
                out[p] = roc_auc_score(y[m], pred[m])
        return out if per_observer else float(np.mean(list(out.values())))

    def within_trial(pred, yy, sel=None):
        ok = tot = 0.0
        for t, idx in by_trial.items():
            if sel is not None and not sel[idx[0]]: continue
            pos, neg = idx[yy[idx] == 1], idx[yy[idx] == 0]
            for i in pos:
                ok += np.sum(pred[i] > pred[neg]) + 0.5 * np.sum(pred[i] == pred[neg]); tot += len(neg)
        return ok / tot

    p_own = lopo(Z)
    p_plus = lopo(np.column_stack([Z, others]))
    p_wt = loco(Z)
    res = dict(feature_set=name, n_features=Z.shape[1], n_trials=len(y), C=C,
               auc_lopo=auc_obs(p_own), auc_within_cv=within_cv(Z),
               auc_ceiling_others=auc_obs(others), auc_metal=auc_obs(F.metal.values),
               auc_lopo_plus_condition=auc_obs(p_plus), auc_loco=auc_obs(p_wt),
               within_trial_auc=within_trial(p_wt, y))
    obs_auc = pd.DataFrame(dict(participant=PIDS))
    obs_auc["auc_own"] = obs_auc.participant.map(auc_obs(p_own, per_observer=True))
    obs_auc["auc_plus_condition"] = obs_auc.participant.map(auc_obs(p_plus, per_observer=True))
    obs_auc["auc_others"] = obs_auc.participant.map(auc_obs(others, per_observer=True))
    obs_auc["auc_metal"] = obs_auc.participant.map(auc_obs(F.metal.values, per_observer=True))
    null = []
    for _ in range(n_perm):
        yp = y.copy()
        for t, idx in by_trial.items():
            yp[idx] = rng.permutation(yp[idx])
        null.append(within_trial(p_wt, yp))
    null = np.array(null) if n_perm else np.array([np.nan])
    res["within_trial_null_mean"] = null.mean(); res["within_trial_null_sd"] = null.std()
    res["within_trial_p"] = (np.sum(null >= res["within_trial_auc"]) + 1) / (n_perm + 1)
    res["within_trial_z"] = (res["within_trial_auc"] - null.mean()) / null.std() if n_perm else np.nan
    sw, sw_obs, sw_wt, p_sw_mean, p_sw_wt_mean = [], [], [], np.zeros(len(y)), np.zeros(len(y))
    for _ in range(n_swap):
        Zs = Z.copy()
        for i in range(len(y)):
            cand = by_trial[trials[i]]; cand = cand[cand != i]
            Zs[i] = Z[rng.choice(cand)]
        p_sw = lopo(Zs); p_sw_mean += p_sw / n_swap
        p_sw_wt = loco(Zs); p_sw_wt_mean += p_sw_wt / n_swap
        sw.append(auc_obs(p_sw)); sw_obs.append(auc_obs(p_sw, per_observer=True)); sw_wt.append(within_trial(p_sw_wt, y))
    if n_swap:
        res["auc_lopo_swapped"] = float(np.mean(sw)); res["auc_lopo_swapped_sd"] = float(np.std(sw))
        res["within_trial_auc_swapped"] = float(np.mean(sw_wt)); res["within_trial_auc_swapped_sd"] = float(np.std(sw_wt))
        obs_auc["auc_swapped"] = obs_auc.participant.map(lambda p: float(np.mean([d[p] for d in sw_obs])))
    for mod in ("hand", "head", "both"):
        sel = F.modality.values == mod
        res[f"auc_lopo_{mod}"] = auc_obs(p_own, sel)
        res[f"auc_metal_{mod}"] = auc_obs(F.metal.values, sel)
        res[f"auc_ceiling_{mod}"] = auc_obs(others, sel)
        res[f"within_trial_auc_{mod}"] = within_trial(p_wt, y, sel)
    return res, pd.DataFrame(dict(participant=pids, trial=trials, resp=y, score=p_own, score_plus=p_plus, score_swapped=p_sw_mean,
                                  score_wt=p_wt, score_wt_swapped=p_sw_wt_mean)), obs_auc


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("feature_set")
    ap.add_argument("--C", type=float, default=0.1)
    ap.add_argument("--pca", type=int, default=None)
    a = ap.parse_args()
    res, scores, obs_auc = evaluate(a.feature_set, C=a.C, pca=a.pca)
    out = RESULTS / "models"; out.mkdir(exist_ok=True)
    tag = a.feature_set + (f"_pca{a.pca}" if a.pca else "") + f"_C{a.C}"
    pd.DataFrame([res]).to_csv(out / f"{tag}.csv", index=False)
    scores.to_csv(out / f"{tag}_scores.csv", index=False)
    obs_auc.to_csv(out / f"{tag}_observer_auc.csv", index=False)
    for k, v in res.items():
        print(f"{k:28s} {v:.3f}" if isinstance(v, float) else f"{k:28s} {v}")
