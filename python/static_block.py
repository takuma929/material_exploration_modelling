"""The static block (trial IDs 1-40): the same statistics on the one image
every observer saw, and the same evaluations as for the moving blocks.

Features: the per-frame means of the 14 object quantities (object crop) and
the 9 relational quantities (object + surround patch) of the single frame;
the frame-difference energy and the summaries of change do not exist without
motion, so the moving blocks are re-evaluated with the same 23 means for
comparison, and a model fitted to the moving videos is applied to the static
image (and the reverse).  The disagreement test is scored
leave-one-stimulus-out as in evaluate.py; in the static block every observer
saw the same image, so the test must give chance.

Writes results/features/static/participant_XXX.csv
       results/models/static_block.csv                  one row per block x model
       results/models/static_block_scores.csv           per-video scores
       results/models/static_within_trial_by_observer.csv
       results/behaviour/static_observer_summary.csv    per observer
       results/behaviour/static_tests.csv               paired tests across the 19 observers
       results/behaviour/block_means.csv                P(metal) and duration by block and level
"""
import sys
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, ttest_rel, ttest_1samp, wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from common import ROOT, RESULTS, FEATURES, STATIC, trial_dirs, frame_paths, context_crop
import extract_image_stats as eis
import extract_relational_stats as ers

eis.CROP = "object"
OUT = FEATURES / "static"
OUT.mkdir(parents=True, exist_ok=True)
C = 0.1
N_PERM = 300
OBJ_Q = ["area", "Y_mean", "Y_sd", "Y_med", "Y_max", "Y_skew", "Y_kurt", "rms_contrast", "hl_frac", "hl_contrast",
         "edge_energy", "dark_frac", "chroma_mean", "chroma_sd"]
QUANT = [f"{q}_mean" for q in OBJ_Q + ers.QUANT]
META = ["participant", "trial", "resp", "metal", "modality", "env", "geom", "n_frames"]

TS = pd.read_csv(ROOT / "data" / "static_trials.csv", dtype={"participant": str})
TM = pd.read_csv(ROOT / "data" / "trials.csv", dtype={"participant": str})
TM["dur_s"] = (TM.t_last_ms - TM.t_first_ms) / 1000
PIDS = sorted(TS.participant.unique())


# ---------------------------------------------------------------- features
def static_features(pid):
    rows = []
    for d, info in trial_dirs(pid, "Cropped", STATIC):
        fp = frame_paths(d)[0]
        rgb, obj = eis.load_patch(d, fp)
        s, _, _ = eis.frame_stats(rgb, obj)
        tp = STATIC / f"participant_{pid}" / "Transparent" / d.name / fp.name
        r = context_crop(str(tp).replace("/Transparent/", "/Color/"), tp)
        q = ers.frame_quantities(r[0].astype(np.float32) / 255, r[1])
        rows.append(dict(participant=pid, trial=info["trial"], **{f"{k}_mean": v for k, v in {**s, **q}.items()}))
    pd.DataFrame(rows).to_csv(OUT / f"participant_{pid}.csv", index=False)
    return len(rows)


if "--features" in sys.argv or not (OUT / f"participant_{PIDS[-1]}.csv").exists():
    for pid in PIDS:
        print(pid, static_features(pid), "trials", flush=True)

FS = pd.concat([pd.read_csv(OUT / f"participant_{p}.csv", dtype={"participant": str}) for p in PIDS], ignore_index=True)
FS = FS.merge(TS[META], on=["participant", "trial"]).reset_index(drop=True)
parts = [pd.read_csv(f, dtype={"participant": str}) for f in sorted((FEATURES / "image_stats_object").glob("participant_*.csv"))]
FM = pd.concat(parts, ignore_index=True).merge(
    pd.concat([pd.read_csv(f, dtype={"participant": str}) for f in sorted((FEATURES / "relational").glob("participant_*.csv"))]),
    on=["participant", "trial"]).merge(TM[META], on=["participant", "trial"]).reset_index(drop=True)


# ---------------------------------------------------------------- helpers (as evaluate.py)
def zscore(F, cols=QUANT):
    return F.groupby("participant")[cols].transform(lambda s: (s - s.mean()) / (s.std() + 1e-12)).values


def conds_of(F):
    return (F.env + "_" + F.geom + "_" + F.metal.astype(str)).values


def fit(X, y):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000)).fit(X, y)


def lopo(Xtr, ytr, ptr, Xte, pte):
    """Leave-one-observer-out: train on all observers but one, score the held-out observer's videos."""
    pred = np.zeros(len(pte))
    for p in PIDS:
        pred[pte == p] = fit(Xtr[ptr != p], ytr[ptr != p]).decision_function(Xte[pte == p])
    return pred


def loco(Z, y, conds):
    """Leave-one-stimulus-out: one model per stimulus, fitted to every video of the other stimuli
    (the scores of the disagreement test, as evaluate.loco)."""
    pred = np.zeros(len(y))
    for c in np.unique(conds):
        te = conds == c
        pred[te] = fit(Z[~te], y[~te]).decision_function(Z[te])
    return pred


def auc_obs(y, pred, p, sel=None, per=False):
    out = {}
    for q in PIDS:
        m = (p == q) if sel is None else (p == q) & sel
        if m.sum() and 0 < y[m].mean() < 1:
            out[q] = roc_auc_score(y[m], pred[m])
    return out if per else float(np.mean(list(out.values())))


def within_cv(X, y, p, seed=0):
    aucs = []
    for q in PIDS:
        idx = np.where(p == q)[0]; pr = np.zeros(len(idx))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(idx, y[idx]):
            pr[te] = fit(X[idx[tr]], y[idx[tr]]).decision_function(X[idx[te]])
        aucs.append(roc_auc_score(y[idx], pr))
    return float(np.mean(aucs))


def within_trial(pred, y, trials, sel=None):
    ok = tot = 0.0
    for t in np.unique(trials):
        idx = np.where(trials == t)[0]
        if sel is not None and not sel[idx[0]]: continue
        pos, neg = idx[y[idx] == 1], idx[y[idx] == 0]
        for i in pos:
            ok += np.sum(pred[i] > pred[neg]) + 0.5 * np.sum(pred[i] == pred[neg]); tot += len(neg)
    return ok / tot if tot else np.nan


def within_trial_null(pred, y, trials, sel=None, n=N_PERM, seed=0):
    rng = np.random.default_rng(seed); out = []
    for _ in range(n):
        yp = y.copy()
        for t in np.unique(trials):
            idx = np.where(trials == t)[0]; yp[idx] = rng.permutation(yp[idx])
        out.append(within_trial(pred, yp, trials, sel))
    return np.array(out)


def within_trial_obs(pred, y, trials, p):
    """Per observer: the AUC over the disagreeing pairs that contain the observer (as observer_within_trial.py)."""
    out = {}
    for q in PIDS:
        ok = tot = 0.0
        for t in np.unique(trials):
            idx = np.where(trials == t)[0]; me = idx[p[idx] == q]
            if len(me) != 1: continue
            me = me[0]; others = idx[(p[idx] != q) & (y[idx] != y[me])]
            if len(others) == 0: continue
            s_me, s_ot = pred[me], pred[others]
            ok += np.sum(s_me > s_ot) + 0.5 * np.sum(s_me == s_ot) if y[me] == 1 else np.sum(s_me < s_ot) + 0.5 * np.sum(s_me == s_ot)
            tot += len(others)
        out[q] = ok / tot
    return pd.Series(out)


def others_of(F):
    return F.groupby("trial").resp.transform(lambda s: (s.sum() - s) / (len(s) - 1)).values


def evaluate(F, label, Z=None):
    Z = zscore(F) if Z is None else Z
    y, p, t = F.resp.values, F.participant.values, F.trial.values
    pred = lopo(Z, y, p, Z, p)
    p_wt = loco(Z, y, conds_of(F))
    others = others_of(F)
    res = dict(block=label, n_features=Z.shape[1], n_videos=len(y),
               auc_lopo=auc_obs(y, pred, p), auc_within_cv=within_cv(Z, y, p), auc_loco=auc_obs(y, p_wt, p),
               auc_ceiling_others=auc_obs(y, others, p), auc_metal=auc_obs(y, F.metal.values, p),
               within_trial_auc=within_trial(p_wt, y, t))
    null = within_trial_null(p_wt, y, t)
    res["within_trial_null_mean"], res["within_trial_null_sd"] = null.mean(), null.std()
    res["within_trial_z"] = (res["within_trial_auc"] - null.mean()) / null.std()
    res["within_trial_p"] = (np.sum(null >= res["within_trial_auc"]) + 1) / (len(null) + 1)
    for mod in sorted(F.modality.unique()):
        sel = F.modality.values == mod
        res[f"auc_lopo_{mod}"] = auc_obs(y, pred, p, sel); res[f"auc_metal_{mod}"] = auc_obs(y, F.metal.values, p, sel)
        res[f"auc_ceiling_{mod}"] = auc_obs(y, others, p, sel); res[f"within_trial_auc_{mod}"] = within_trial(p_wt, y, t, sel)
    obs = pd.DataFrame(dict(participant=PIDS))
    for k, v in (("auc_own", auc_obs(y, pred, p, per=True)), ("auc_others", auc_obs(y, others, p, per=True)),
                 ("auc_metal", auc_obs(y, F.metal.values, p, per=True)), ("wt", within_trial_obs(p_wt, y, t, p))):
        obs[k] = obs.participant.map(v)
    return res, pd.DataFrame(dict(block=label, participant=p, trial=t, resp=y, score=pred, score_wt=p_wt)), obs


rows, scores, obs_tables = [], [], {}
for F, label in ((FS, "static_23means"), (FM, "moving_23means")):
    r, s, o = evaluate(F, label); rows.append(r); scores.append(s); obs_tables[label] = o
    print(label, {k: round(v, 3) for k, v in r.items() if isinstance(v, float)}, flush=True)

# viewing duration alone in the static block (the log of the number of frames, as the duration control)
ys, ps, ts = FS.resp.values, FS.participant.values, FS.trial.values
Zd = zscore(FS.assign(log_dur=np.log(FS.n_frames)), ["log_dur"])
p_d = loco(Zd, ys, conds_of(FS)); null = within_trial_null(p_d, ys, ts)
rows.append(dict(block="static_duration_only", n_features=1, n_videos=len(ys), auc_lopo=auc_obs(ys, lopo(Zd, ys, ps, Zd, ps), ps),
                 within_trial_auc=within_trial(p_d, ys, ts), within_trial_null_mean=null.mean(), within_trial_null_sd=null.std(),
                 within_trial_z=(within_trial(p_d, ys, ts) - null.mean()) / null.std()))
wt_obs = pd.DataFrame(dict(participant=PIDS, stats=obs_tables["static_23means"].wt.values,
                           duration_only=within_trial_obs(p_d, ys, ts, ps).reindex(PIDS).values))
wt_obs.to_csv(RESULTS / "models" / "static_within_trial_by_observer.csv", index=False)

# transfer across blocks: a model fitted to the other 18 observers' moving videos scores the
# held-out observer's static image, and the reverse
ZS, ZM = zscore(FS), zscore(FM)
ym, pm, tm = FM.resp.values, FM.participant.values, FM.trial.values
pred = lopo(ZM, ym, pm, ZS, ps)
rows.append(dict(block="static_from_moving", n_features=ZS.shape[1], n_videos=len(ys), auc_lopo=auc_obs(ys, pred, ps)))
obs_tables["static_from_moving"] = auc_obs(ys, pred, ps, per=True)
pred = lopo(ZS, ys, ps, ZM, pm)
r = dict(block="moving_from_static", n_features=ZM.shape[1], n_videos=len(ym), auc_lopo=auc_obs(ym, pred, pm))
for mod in ("hand", "head", "both"):
    r[f"auc_lopo_{mod}"] = auc_obs(ym, pred, pm, FM.modality.values == mod)
rows.append(r)

R = pd.DataFrame(rows)
R.to_csv(RESULTS / "models" / "static_block.csv", index=False)
pd.concat(scores).to_csv(RESULTS / "models" / "static_block_scores.csv", index=False)
print(R.round(3).T.to_string())

# ---------------------------------------------------------------- behaviour, all four blocks
B = RESULTS / "behaviour"
OS = pd.read_csv(B / "observer_summary.csv", dtype={"participant": str})
o = obs_tables["static_23means"]
S = pd.DataFrame(dict(participant=PIDS, auc_stats_static=o.auc_own.values, auc_others_static=o.auc_others.values,
                      auc_metal_static=o.auc_metal.values, wt_stats_static=o.wt.values))
S["auc_stats_static_from_moving"] = S.participant.map(obs_tables["static_from_moving"])
Rs = TS.pivot(index="trial", columns="participant", values="resp")
S["auc_all_static"] = [roc_auc_score(Rs[p], Rs.mean(axis=1)) for p in S.participant]      # the observer included: upper bound of the noise ceiling
S["p_metal_static"] = S.participant.map(TS.groupby("participant").resp.mean())
S["dur_static"] = S.participant.map(TS.groupby("participant").dur_s.mean())
S.to_csv(B / "static_observer_summary.csv", index=False)

TA = pd.concat([TS[META + ["dur_s"]], TM[META + ["dur_s"]]], ignore_index=True)
pm_ = TA.groupby(["participant", "modality"]).resp.mean().unstack()
du_ = TA.groupby(["participant", "modality"]).dur_s.mean().unstack()
lv = TA.groupby(["participant", "modality", "metal"]).resp.mean().unstack("metal")
bm = pd.concat([TA.groupby(["modality", "metal"]).resp.mean().rename("p_metal"),
                TA.groupby(["modality", "metal"]).dur_s.mean().rename("dur_s")], axis=1).reset_index()
bm.to_csv(B / "block_means.csv", index=False)

tests = []
def paired(name, a, b):
    t = ttest_rel(a, b); w = wilcoxon(a, b)
    tests.append(dict(contrast=name, mean_a=float(np.mean(a)), mean_b=float(np.mean(b)), t=float(t.statistic), df=len(a) - 1, p=float(t.pvalue),
                      wilcoxon_p=float(w.pvalue), n_positive=int(np.sum(np.asarray(a) > np.asarray(b)))))
moving = TM.groupby("participant")
for mod in ("hand", "head", "both"):
    paired(f"auc_metal_static_vs_{mod}", S.auc_metal_static, OS[f"auc_metal_{mod}"])
    paired(f"auc_others_static_vs_{mod}", S.auc_others_static, OS[f"auc_others_{mod}"])
    paired(f"dur_static_vs_{mod}", du_["static"], du_[mod])
    paired(f"p_metal_static_vs_{mod}", pm_["static"], pm_[mod])
paired("auc_metal_static_vs_moving", S.auc_metal_static, OS.auc_metal_level)
paired("auc_others_static_vs_moving", S.auc_others_static, OS.auc_other_observers)
paired("dur_static_vs_moving", du_["static"], moving.dur_s.mean().reindex(PIDS))
paired("p_metal_static_vs_moving", pm_["static"], moving.resp.mean().reindex(PIDS))
for level in (0.0, 1.0):
    k = int(level)
    a = lv.xs("static", level="modality")[level].reindex(PIDS)
    b = TM[TM.metal == level].groupby("participant").resp.mean().reindex(PIDS)
    paired(f"p_metal{k}_static_vs_moving", a, b)
paired("auc_stats_static_vs_others_static", S.auc_stats_static, S.auc_others_static)
paired("auc_stats_static_vs_metal_static", S.auc_stats_static, S.auc_metal_static)
t = ttest_1samp(S.wt_stats_static, 0.5)
tests.append(dict(contrast="wt_stats_static_vs_chance", mean_a=float(S.wt_stats_static.mean()), mean_b=0.5, t=float(t.statistic), df=len(S) - 1,
                  p=float(t.pvalue), wilcoxon_p=float(wilcoxon(S.wt_stats_static - 0.5).pvalue), n_positive=int((S.wt_stats_static > 0.5).sum())))
for name, d in (("resp", pm_), ("dur", du_)):
    fr = friedmanchisquare(d["static"], d["hand"], d["head"], d["both"])
    tests.append(dict(contrast=f"friedman4_{name}", mean_a=float(fr.statistic), mean_b=float("nan"), t=float("nan"), df=3, p=float(fr.pvalue),
                      wilcoxon_p=float("nan"), n_positive=0))
Tt = pd.DataFrame(tests)
Tt.to_csv(B / "static_tests.csv", index=False)
print("\nP(metal) by block:", pm_.mean().round(3).to_dict())
print("duration (s) by block, mean:", du_.mean().round(2).to_dict(), " median of trials:", TA.groupby("modality").dur_s.median().round(2).to_dict())
print("\nP(metal) by physical metallicity and block:\n", bm.pivot(index="modality", columns="metal", values="p_metal").round(3).to_string())
print("\n", Tt.round(4).to_string(index=False))
