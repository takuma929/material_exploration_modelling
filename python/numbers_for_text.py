"""Collect every number quoted in the paper into results/numbers.json.

Reads the tables written by the other scripts (nothing is computed from the
frames here) and writes one JSON file whose keys are the placeholders of the
manuscript.  The main groups of keys:

  stats_*                               the main model
                                        (leave-one-observer-out AUC, by condition; the
                                        disagreement test and its permutation null)
  or_*, ostats_*, rel_*, flow_*         the 100 image statistics without motion, and each
                                        family alone (number of statistics, AUC)
  auc_ceiling_*, ceil_tests, sd_*       the ceiling and the reference predictors, with
                                        paired tests of the own-video model against them
  cond_tests                            the own-video AUC compared between the four conditions
  tests, wt_obs                         observer-level tests of observer_within_trial.py
  wtb_*, wts_*, wtd_*, wtl_*            the disagreement test by environment and shape, by
                                        the distance between the two videos, by the score
                                        difference, and by physical metallicity
  pca_*                                 the statistics reduced to k principal components
  sg, sg_*                              single statistics for the group and per observer
  st_*, stfm_*, mvfs_*, pm_*, dur_*     the condition without exploration and the behaviour
                                        of the four conditions
  ill_*, pair_*                         the example videos of Figures 2 and 7
  slide, slide_tests                    the disagreement test on a 2-s window slid back from the response
  n_frames_*, dur_*, n_pairs, ...       counts and durations from the trial table

Run after every other analysis script (see run_pipeline.sh).
"""
import json
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, friedmanchisquare
from common import ROOT, RESULTS, MAIN_FEATURES

T = pd.read_csv(ROOT / "data" / "trials.csv", dtype={"participant": str})
M = pd.read_csv(RESULTS / "summary" / "model_summary.csv")
A = pd.read_csv(RESULTS / "summary" / "per_observer_auc.csv", dtype={"participant": str})
D = pd.read_csv(RESULTS / "models" / "duration_control.csv")
Ab = pd.read_csv(RESULTS / "models" / "feature_ablations.csv")
Wt = pd.read_csv(RESULTS / "models" / "within_trial_features.csv")
n = {}


def row(fs, C):
    r = M[(M.feature_set == fs) & (np.isclose(M.C, C))]
    assert len(r) == 1, fs
    return r.iloc[0]


# ---------------------------------------------------------------- the models (evaluate.py)
for key, fs, C in [("stats", MAIN_FEATURES, 0.1), ("or", "object_relational", 0.1),
                   ("ostats", "image_stats_object", 0.1), ("rel", "relational", 0.1), ("flow", "flow", 0.1)]:
    if not ((M.feature_set == fs) & np.isclose(M.C, C)).any():
        continue
    r = row(fs, C)
    for c in ("auc_lopo", "auc_within_cv", "auc_lopo_plus_condition", "within_trial_auc", "within_trial_null_mean",
              "within_trial_null_sd", "within_trial_z", "within_trial_p", "auc_lopo_swapped", "auc_lopo_swapped_sd",
              "within_trial_auc_swapped", "within_trial_auc_swapped_sd",
              "auc_lopo_hand", "auc_lopo_head", "auc_lopo_both", "within_trial_auc_hand", "within_trial_auc_head",
              "within_trial_auc_both", "n_features"):
        n[f"{key}_{c}"] = float(r[c]) if c in r.index and pd.notna(r[c]) else float("nan")
r = row(MAIN_FEATURES, 0.1)
for c in ("auc_ceiling_others", "auc_metal", "auc_ceiling_hand", "auc_ceiling_head", "auc_ceiling_both",
          "auc_metal_hand", "auc_metal_head", "auc_metal_both"):
    n[c] = float(r[c])
n["flow_n_quant"] = 6                                       # per-frame flow quantities computed (extract_flow_cues.py)
n["flow_dropped"] = int(24 - n.get("flow_n_features", 24))   # flow statistics dropped because undefined in > 1% of the videos

# ---------------------------------------------------------------- the ceiling and the reference predictors (collect_results.py)
Au = A[A.predictor == "all_observers"].set_index("participant"); Al = A[A.predictor == "other_observers"].set_index("participant")
Ao = A[A.predictor == "image_stats"].set_index("participant")
n["auc_ceiling_upper"] = float(Au.auc_all.mean()); n["sd_ceiling_upper"] = float(Au.auc_all.std())
for mod in ("hand", "head", "both"):
    n[f"auc_ceiling_upper_{mod}"] = float(Au[f"auc_{mod}"].mean())
n["ceil_tests"] = {}
for name, a, b in (("own_vs_upper", Ao.auc_all, Au.auc_all.loc[Ao.index]), ("own_vs_lower", Ao.auc_all, Al.auc_all.loc[Ao.index]),
                   ("own_vs_upper_both", Ao.auc_both, Au.auc_both.loc[Ao.index]), ("own_vs_lower_both", Ao.auc_both, Al.auc_both.loc[Ao.index]),
                   ("own_vs_upper_head", Ao.auc_head, Au.auc_head.loc[Ao.index]), ("own_vs_upper_hand", Ao.auc_hand, Au.auc_hand.loc[Ao.index])):
    t = ttest_rel(a, b)
    n["ceil_tests"][name] = dict(mean_a=float(a.mean()), mean_b=float(b.mean()), t=float(t.statistic), df=int(len(a) - 1), p=float(t.pvalue),
                                 n_positive=int((a.values > b.values).sum()))
for pred in ("image_stats", "metal_parameter", "other_observers", "image_stats_object", "relational", "flow", "object_relational"):
    if (A.predictor == pred).any():
        n[f"sd_{pred}"] = float(A[A.predictor == pred].auc_all.std())

# ---------------------------------------------------------------- duration control, ablations, single statistics of the disagreement test (within_trial_features.py)
for d in D.itertuples():
    n[f"dur_{d.modality}_full"] = d.full; n[f"dur_{d.modality}_no_duration"] = d.no_duration; n[f"dur_{d.modality}_only"] = d.duration_only
for a in Ab.itertuples():
    n[f"abl_{a.model}"] = a.within_trial_auc; n[f"abl_{a.model}_n"] = int(a.n_features)
Wt = Wt[Wt.feature != "log_dur"].set_index("feature")
n["single"] = Wt.within_trial_auc.round(3).to_dict()

# ---------------------------------------------------------------- observer-level tests (observer_within_trial.py)
tf = RESULTS / "models" / "within_trial_tests.csv"
if tf.exists():
    Tt = pd.read_csv(tf)
    n["tests"] = {r.contrast.replace(" vs 0.5", " vs chance").replace(" vs ", "_vs_").replace(" ", "_"):
                  dict(mean_a=r.mean_a, mean_b=r.mean_b, se_a=r.se_a, t=r.t, df=int(r.df), p=r.p, n_positive=int(r.n_positive)) for r in Tt.itertuples()}
    Wb = pd.read_csv(RESULTS / "models" / "within_trial_by_observer.csv")
    n["wt_obs"] = {c: dict(mean=float(Wb[c].mean()), se=float(Wb[c].std() / np.sqrt(len(Wb))), n_above=int((Wb[c] > 0.5).sum())) for c in Wb.columns if c != "participant"}
    n["wt_obs_stats_min"] = float(Wb.stats.min()); n["wt_obs_stats_max"] = float(Wb.stats.max())
gf = RESULTS / "models" / "regularization_grid.csv"
if gf.exists():      # regularization_grid.py
    for r in pd.read_csv(gf).itertuples():
        n[f"grid_{str(r.C).replace('.', 'p')}_lopo"] = r.auc_lopo; n[f"grid_{str(r.C).replace('.', 'p')}_wt"] = r.within_trial_auc

# ---------------------------------------------------------------- the trials: durations, frame counts, responses by condition
dur = (T.t_last_ms - T.t_first_ms) / 1000
n["n_trials_lt2s"] = int((dur < 2).sum()); n["pct_trials_lt2s"] = float(100 * (dur < 2).mean())
n["interval_max_ms"] = float((dur[T.n_frames > 150] * 1000 / 150).max())        # spacing of the 150 sampled frames in the longest trial
n["interval_p95_ms"] = float((dur[T.n_frames > 150] * 1000 / 150).quantile(0.95))
n["frames_per_s_longest"] = 1000 / n["interval_max_ms"]
n["n_frames_min"] = int(T.n_frames.min()); n["n_frames_max"] = int(T.n_frames.max()); n["n_frames_med"] = int(T.n_frames.median())
n["dur_min"] = float(dur.min()); n["dur_max"] = float(dur.max()); n["dur_med"] = float(dur.median()); n["dur_mean"] = float(dur.mean())
n["frame_spacing_med_ms"] = float((dur * 1000 / (T.n_frames - 1)).median())
n["pct_trials_le150"] = float((T.n_frames <= 150).mean() * 100)
n["pct_trials_le64"] = float((T.n_frames <= 64).mean() * 100)
n["n_pairs"] = int(sum(len(g[g.resp == 1]) * len(g[g.resp == 0]) for _, g in T.groupby("trial")))   # disagreeing pairs
pm = T.groupby(["participant", "modality"]).resp.mean().unstack()
fr = friedmanchisquare(pm["hand"], pm["head"], pm["both"]); n["block_resp_friedman_p"] = float(fr.pvalue); n["block_resp_friedman_chi2"] = float(fr.statistic)
pd_ = T.assign(dur=dur).groupby(["participant", "modality"]).dur.mean().unstack()
fr = friedmanchisquare(pd_["hand"], pd_["head"], pd_["both"]); n["block_dur_friedman_p"] = float(fr.pvalue); n["block_dur_friedman_chi2"] = float(fr.statistic)
n["block_resp_means"] = {k: float(v) for k, v in pm.mean().items()}
oa = RESULTS / "models" / f"{MAIN_FEATURES}_C0.1_observer_auc.csv"
if oa.exists():
    O = pd.read_csv(oa)
    n["se_auc_own"] = float(O.auc_own.std() / np.sqrt(len(O))); n["se_auc_swapped"] = float(O.auc_swapped.std() / np.sqrt(len(O))) if "auc_swapped" in O else float("nan")

# ---------------------------------------------------------------- the statistics reduced to k principal components (pca_sweep.py; Figure 4a)
ps = RESULTS / "models" / "pca_sweep.csv"
if ps.exists():
    for _, r in pd.read_csv(ps).iterrows():
        n[f"pca_auc_k{int(r.k)}"] = float(r.auc_mean); n[f"pca_var_k{int(r.k)}"] = float(r.var_explained)

# ---------------------------------------------------------------- the disagreement test split in several ways (Figure 7)
bd = RESULTS / "models" / "within_trial_by_difference.csv"
if bd.exists():      # within_trial_by_difference.py: by the absolute difference in predicted metallicity, and by physical metallicity
    Bd = pd.read_csv(bd)
    for r in Bd.itertuples():
        for c in ("lo", "hi", "d_median", "n", "within_trial_auc", "null_mean", "null_sd", "z", "p",
                  "swapped_auc_all", "swapped_auc_own_split", "swapped_auc_swapped_split", "n_swapped_split"):
            if c in Bd.columns: n[f"wtd_q{r.quartile}_{c}"] = float(getattr(r, c))
    if "swapped_auc_swapped_split" in Bd.columns:
        n["wtd_sw_min"] = float(Bd.swapped_auc_swapped_split.min()); n["wtd_sw_max"] = float(Bd.swapped_auc_swapped_split.max())
        n["wtd_sw_all"] = float(Bd.swapped_auc_all.iloc[0])
    Bl = pd.read_csv(RESULTS / "models" / "within_trial_by_level.csv")
    for r in Bl.itertuples():
        n[f"wtl_{int(round(r.metal * 100))}_auc"] = float(r.within_trial_auc); n[f"wtl_{int(round(r.metal * 100))}_n"] = int(r.n)
    n["wtl_min"] = float(Bl.within_trial_auc.min()); n["wtl_max"] = float(Bl.within_trial_auc.max())
bs_ = RESULTS / "models" / "within_trial_by_similarity.csv"
if bs_.exists():      # within_trial_by_similarity.py: by the distance between the two videos in the space of the statistics (Figure 7c)
    Bs = pd.read_csv(bs_)
    for r in Bs.itertuples():
        for c in ("lo", "hi", "d_median", "n", "within_trial_auc", "null_mean", "null_sd", "z", "p", "score_diff_median"):
            n[f"wts_q{r.quartile}_{c}"] = float(getattr(r, c))
    n["wts_n_components"] = int(Bs.n_components.iloc[0]); n["wts_var_explained"] = float(Bs.var_explained.iloc[0])
bb_ = RESULTS / "models" / "within_trial_breakdown.csv"
if bb_.exists():      # within_trial_breakdown.py: all pairs, by environment and by shape (Figure 7b)
    Bb = pd.read_csv(bb_)
    for r in Bb.itertuples():
        for c in ("n_pairs", "within_trial_auc", "null_mean", "null_sd", "z", "p", "obs_mean", "obs_se", "n_obs_above"):
            n[f"wtb_{r.group}_{c}"] = float(getattr(r, c))
    env = Bb[Bb.group.str.startswith("env_")]; geo = Bb[Bb.group.str.startswith("geom_")]
    n["wtb_env_min"] = float(env.within_trial_auc.min()); n["wtb_env_max"] = float(env.within_trial_auc.max())
    n["wtb_env_min_name"] = env.group[env.within_trial_auc.idxmin()][4:]; n["wtb_env_max_name"] = env.group[env.within_trial_auc.idxmax()][4:]
    n["wtb_env_z_min"] = float(env.z.min()); n["wtb_geom_z_min"] = float(geo.z.min())

# ---------------------------------------------------------------- the own-video AUC compared between the four conditions (Figure 4b)
S_ = pd.read_csv(RESULTS / "behaviour" / "static_observer_summary.csv", dtype={"participant": str})
a_ = Ao.sort_index(); s_ = S_.set_index("participant").sort_index()
D_ = pd.DataFrame({"head": a_.auc_head, "hand": a_.auc_hand, "both": a_.auc_both, "static": s_.auc_stats_static})
fr = friedmanchisquare(D_["head"], D_["hand"], D_["both"], D_["static"])
n["cond_tests"] = {"friedman4_stats": {"chi2": float(fr.statistic), "p": float(fr.pvalue), "df": 3}}
for x_, y_ in [("both", "head"), ("both", "static"), ("hand", "head"), ("hand", "static"), ("head", "static"), ("both", "hand")]:
    t_ = ttest_rel(D_[x_], D_[y_])
    n["cond_tests"][f"stats_{x_}_vs_{y_}"] = {"t": float(t_.statistic), "p": float(t_.pvalue), "df": len(D_) - 1, "n_positive": int((D_[x_] > D_[y_]).sum())}

# ---------------------------------------------------------------- the example video of Figure 2 (save_model_illustration.py, save_example_frames.py)
mi = ROOT / "data" / "example_frames" / "model_illustration.csv"
if mi.exists():
    Mi = pd.read_csv(mi).set_index("feature").w
    for k in ("observer", "trial", "n_metal", "n_glass", "n_pairs", "metal_higher"):
        n[f"ill_{k}"] = int(Mi[k])
    n["ill_score"] = float(Mi["score"]); n["ill_auc"] = float(Mi["auc"])
    tr_ = T[(T.participant == f"{int(Mi['observer']):03d}") & (T.trial == int(Mi["trial"]))].iloc[0]
    n["ill_env"] = tr_.env; n["ill_geom"] = tr_.geom.lower(); n["ill_metal"] = float(tr_.metal)
    se = ROOT / "data" / "example_frames" / "score_examples.csv"
    if se.exists():
        for _, r in pd.read_csv(se).iterrows():      # the lowest- and highest-scoring videos of the observer (Figure 2c)
            n[f"ill_{r.role}_score"] = float(r.score); n[f"ill_{r.role}_trial"] = int(r.trial)
            n[f"ill_{r.role}_env"] = r.env; n[f"ill_{r.role}_geom"] = r.geom.lower(); n[f"ill_{r.role}_metal"] = float(r.metal)
            n[f"ill_{r.role}_resp"] = ["glass", "metal"][int(r.resp)]

# ---------------------------------------------------------------- single statistics against the responses (single_statistics.py; Figures 5 and 6)
sg = RESULTS / "models" / "single_statistics.csv"
if sg.exists():
    Sg = pd.read_csv(sg)
    n["sg"] = {r.feature: float(r.auc) for r in Sg.itertuples()}
    n["sg_null_sd"] = float(Sg.null_sd.mean())
    top = pd.concat([Sg.sort_values("auc", ascending=False).head(8), Sg.sort_values("auc").head(8)])   # Figure 5: the 8 most "metal" and the 8 most "glass"
    So = pd.read_csv(RESULTS / "models" / "single_statistics_by_observer.csv", dtype={"participant": str}).set_index("participant")
    feats = list(So.columns)
    sign = np.sign(Sg.set_index("feature").auc - 0.5)
    aligned = 0.5 + (So[feats] - 0.5) * sign[feats].values           # AUC in the group's direction of each statistic
    best = aligned.idxmax(axis=1); vc = best.value_counts()
    n["sg_n_features"] = int(len(feats)); n["sg_n_distinct_best"] = int(len(vc)); n["sg_best_max_n"] = int(vc.iloc[0]); n["sg_best_top"] = str(vc.index[0])
    n["sg_best_min"] = float(aligned.max(axis=1).min()); n["sg_best_max"] = float(aligned.max(axis=1).max())
    Oa_ = pd.read_csv(RESULTS / "models" / f"{MAIN_FEATURES}_C0.1_observer_auc.csv", dtype={"participant": str}).set_index("participant")
    n["sg_n_best_above_model"] = int((aligned.max(axis=1) > Oa_.auc_own.loc[aligned.index]).sum())
    n_dir16 = (aligned[list(top.feature)] > 0.5).sum(axis=1)          # of the 16 statistics of Figure 5, how many point the group's way for each observer
    # the best statistic in each direction per observer (raw AUC: > 0.5 metal direction, < 0.5 glass direction; the large symbols of Figure 6)
    bm = So[feats].idxmax(axis=1); bg = So[feats].idxmin(axis=1)
    n["sg_bm_n_distinct"] = int(bm.nunique()); n["sg_bm_top"] = str(bm.value_counts().index[0]); n["sg_bm_top_n"] = int(bm.value_counts().iloc[0])
    n["sg_bm_min"] = float(So[feats].max(axis=1).min()); n["sg_bm_max"] = float(So[feats].max(axis=1).max())
    n["sg_bg_n_distinct"] = int(bg.nunique()); n["sg_bg_top"] = str(bg.value_counts().index[0]); n["sg_bg_top_n"] = int(bg.value_counts().iloc[0])
    n["sg_bg_min"] = float(1 - So[feats].min(axis=1).max()); n["sg_bg_max"] = float(1 - So[feats].min(axis=1).min())
    n["sg_bm_n_above_model"] = int((So[feats].max(axis=1) > Oa_.auc_own.loc[So.index]).sum())
    n["sg_n_dir16_min"] = int(n_dir16.min()); n["sg_n_dir16_median"] = float(n_dir16.median())
    n["sg_top_n_same_min"] = int(top.n_same_direction.min())
    # the best single statistic of each observer in either direction against the model that combines all statistics,
    # and how many statistics come within 0.05 of the best one
    either = np.maximum(So[feats], 1 - So[feats])
    n["sg_n_model_above_best"] = int((Oa_.auc_own.loc[So.index] > either.max(axis=1)).sum())
    near = (either.ge(either.max(axis=1) - 0.05, axis=0)).sum(axis=1)
    n["sg_n_near_best_median"] = float(near.median()); n["sg_n_near_best_min"] = int(near.min()); n["sg_n_near_best_max"] = int(near.max())

# ---------------------------------------------------------------- the condition without exploration (static_block.py; Figures 3 and 4b)
sb = RESULTS / "models" / "static_block.csv"
if sb.exists():
    Sb = pd.read_csv(sb).set_index("block")
    for key, blk in (("st", "static_23means"), ("mv23", "moving_23means"), ("stdur", "static_duration_only"),
                     ("stfm", "static_from_moving"), ("mvfs", "moving_from_static")):
        for c in Sb.columns:
            if pd.notna(Sb.loc[blk, c]):
                n[f"{key}_{c}"] = float(Sb.loc[blk, c])
    n["st_auc_ceiling_upper"] = float(S_.auc_all_static.mean())
    t = ttest_rel(S_.auc_stats_static, S_.auc_all_static); n["st_tests_own_vs_upper"] = dict(t=float(t.statistic), df=int(len(S_) - 1), p=float(t.pvalue))
    for c in ("auc_stats_static", "auc_others_static", "auc_metal_static", "auc_stats_static_from_moving", "wt_stats_static"):
        n[f"sd_{c}"] = float(S_[c].std()); n[f"se_{c}"] = float(S_[c].std() / np.sqrt(len(S_))); n[f"mean_{c}"] = float(S_[c].mean())
    St = pd.read_csv(RESULTS / "behaviour" / "static_tests.csv")
    n["st_tests"] = {r.contrast: dict(mean_a=r.mean_a, mean_b=r.mean_b, t=r.t, df=int(r.df), p=r.p, wilcoxon_p=r.wilcoxon_p, n_positive=int(r.n_positive)) for r in St.itertuples()}
    Bm = pd.read_csv(RESULTS / "behaviour" / "block_means.csv")
    for r in Bm.itertuples():
        n[f"pm_{r.modality}_{int(round(r.metal * 100))}"] = float(r.p_metal); n[f"dur_{r.modality}_{int(round(r.metal * 100))}"] = float(r.dur_s)
    Ts = pd.read_csv(ROOT / "data" / "static_trials.csv", dtype={"participant": str})
    n["st_dur_med"] = float(Ts.dur_s.median()); n["st_dur_mean"] = float(Ts.dur_s.mean()); n["st_dur_min"] = float(Ts.dur_s.min()); n["st_dur_max"] = float(Ts.dur_s.max())
    n["st_p_metal"] = float(Ts.resp.mean()); n["st_n_videos"] = int(len(Ts))
    n["st_n_pairs"] = int(sum(len(g[g.resp == 1]) * len(g[g.resp == 0]) for _, g in Ts.groupby("trial")))
    n["mv_p_metal"] = float(T.resp.mean())
    for lvl in (0, 25, 50, 75, 100):
        n[f"pm_moving_{lvl}"] = float(T[T.metal == lvl / 100].resp.mean())
    n["mv_dur_med"] = float(dur.median())
    Ss = pd.read_csv(RESULTS / "models" / "static_block_scores.csv", dtype={"participant": str}); Ss = Ss[Ss.block == "static_23means"]
    Sm = pd.read_csv(RESULTS / "models" / f"{MAIN_FEATURES}_C0.1_scores.csv", dtype={"participant": str})
    n["st_score_sd_within"] = float(Ss.groupby("trial").score_wt.std().mean()); n["st_score_sd_between"] = float(Ss.groupby("trial").score_wt.mean().std())
    n["mv_score_sd_within"] = float(Sm.groupby("trial").score_wt.std().mean()); n["mv_score_sd_between"] = float(Sm.groupby("trial").score_wt.mean().std())

# ---------------------------------------------------------------- the example pair of Figure 7a (save_example_frames.py pair)
pf = ROOT / "data" / "example_frames" / "pair.csv"
if pf.exists():
    P = pd.read_csv(pf, dtype={"participant": str})
    for r in P.itertuples():
        n[f"pair_{r.role}_obs"] = int(r.participant); n[f"pair_{r.role}_score"] = float(r.score)
        n[f"pair_{r.role}_dur"] = float(T[(T.participant == r.participant) & (T.folder == r.folder)].iloc[0].t_last_ms / 1000)
    parts = P.folder.iloc[0].split("_")
    n["pair_trial"] = int(parts[0]); n["pair_env"] = parts[1]; n["pair_geom"] = parts[2].lower(); n["pair_metal"] = parts[3]; n["pair_metal_num"] = float(parts[3])

# ---------------------------------------------------------------- a 2-s window slid back from the response (sliding_window.py)
# slide[window]: the disagreement test on the statistics of one window (pre0to2s = the last 2 s, pre2to4s = from 4 to 2 s
# before the response, ...), with the videos that cover it; pct_videos: their share of all videos;
# slide_tests: the peak window (from 3 to 1 s before the response) against its neighbors, paired across observers:
# against the window before it, against the last window, and against the last window on the videos of the peak window only
sf = RESULTS / "models" / "sliding_window.csv"
if sf.exists():
    Sw = pd.read_csv(sf)
    n["slide"] = {r.window: dict(wt=float(r.within_trial_auc), null_mean=float(r.null_mean), null_sd=float(r.null_sd), z=float(r.z), p=float(r.p),
                                 n_videos=int(r.n_videos), pct_videos=float(100 * r.n_videos / len(T)), n_pairs=int(r.n_pairs),
                                 obs_mean=float(r.obs_mean), obs_se=float(r.obs_se), n_obs=int(r.n_obs)) for r in Sw[Sw.min_duration_s == 0].itertuples()}
    Wo = pd.read_csv(RESULTS / "models" / "sliding_window_by_observer.csv", dtype={"participant": str}).set_index("participant")
    n["slide_tests"] = {}
    for key, a, b in [("peak_vs_before", "pre1to3s", "pre2to4s"), ("peak_vs_last", "pre1to3s", "pre0to2s"), ("peak_vs_last_matched", "pre1to3s", "pre0to2s_min3")]:
        t = ttest_rel(Wo[a], Wo[b])
        n["slide_tests"][key] = dict(mean_a=float(Wo[a].mean()), mean_b=float(Wo[b].mean()), t=float(t.statistic), df=int(len(Wo) - 1), p=float(t.pvalue))

# ---------------------------------------------------------------- effect sizes
# Cohen's d of every paired or one-sample t-test: the mean difference over the standard deviation of the
# differences, d = t / sqrt(N) with N = df + 1 observers; Kendall's W of the Friedman tests, W = chi2 / (N (k - 1))
def add_d(x):
    if isinstance(x, dict):
        if "t" in x and "df" in x and x["t"] == x["t"]:
            x["d"] = float(x["t"] / np.sqrt(x["df"] + 1))
        for v in x.values():
            add_d(v)
add_d(n)
N_OBS = T.participant.nunique()
n["block_resp_friedman_W"] = n["block_resp_friedman_chi2"] / (N_OBS * 2)
n["block_dur_friedman_W"] = n["block_dur_friedman_chi2"] / (N_OBS * 2)
n["cond_tests"]["friedman4_stats"]["W"] = n["cond_tests"]["friedman4_stats"]["chi2"] / (N_OBS * 3)

json.dump(n, open(RESULTS / "numbers.json", "w"), indent=1)
for k, v in n.items():
    if k not in ("single", "sg"):
        print(f"{k:36s} {v:.3f}" if isinstance(v, float) else f"{k:36s} {v}")
