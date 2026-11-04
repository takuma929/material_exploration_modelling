"""Example frames for the figures (data/example_frames/).

  view_<env>_<geom>_m<0|1>.png   full first-person view (720 px) of observer 005
  ctx_<env>_<geom>_m<0|1>.png    the model's input: object + surround patch (224 px)
  boxes.csv                      crop box of each view, in pixels
  timecourse_<k>_<t>s.png / timecourse_ctx_<k>_<t>s.png   one trial over time
  pair_<metal|glass>_<k>_<t>s.png, pair.csv   two observers, same trial ID, different responses (Figure 7a;
                                    needs the scores of the main model, results/models/)
  timecourse_stats.csv, example_object.png, example_patch[_mask].png, consecutive_<k>.png   for Figure A1
  score_<low|high>_ctx_<k>_<t>s.png, score_examples.csv   the lowest- and highest-scoring videos of observer 005 (Figure 2c)
  trial_<low|high>_ctx_<k>_<t>s.png, trial_examples.csv   the lowest- and highest-scoring videos of trial 123 (Figure 2d)
"""
import sys
import numpy as np
import pandas as pd
from PIL import Image
from common import ROOT, IMGS, RESULTS, trial_dirs, frame_paths, context_crop, object_bbox, CROP_MARGIN

OUT = ROOT / "data" / "example_frames"
OUT.mkdir(exist_ok=True)
PID = "005"


def crop_box(tp):
    a = np.asarray(Image.open(tp).split()[-1])
    x0, y0, x1, y1 = object_bbox(a)
    side = int(np.ceil(max(x1 - x0, y1 - y0) * (1 + 2 * CROP_MARGIN)))
    H, W = a.shape
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    L = int(round(min(max(cx - side / 2, 0), W - side)))
    T = int(round(min(max(cy - side / 2, 0), H - side)))
    return L, T, side


def save_pair(frames, stem, n):
    for k, i in enumerate(np.linspace(0, len(frames) - 1, n).round().astype(int)):
        fp = frames[i]
        t = (float(fp.stem) - float(frames[0].stem)) / 1000
        rgb, _ = context_crop(str(fp).replace("/Transparent/", "/Color/"), fp)
        Image.fromarray(rgb).save(OUT / f"{stem}_{k}_{t:.2f}s.png")


def stimuli():
    rows = []
    tds = trial_dirs(PID)
    for env in ("Check", "Gray", "Sky", "Studio"):
        for geom in ("Bumpy", "Smooth"):
            for m in (0, 1):
                cands = [(d, i) for d, i in tds if i["env"] == env and i["geom"] == geom and i["metal"] == float(m) and i["modality"] == "both"]
                d, info = cands[0]
                fr = frame_paths(d)
                fp = fr[len(fr) // 3]
                Image.open(str(fp).replace("/Transparent/", "/Color/")).convert("RGB").save(OUT / f"view_{env}_{geom}_m{m}.png")
                rgb, _ = context_crop(str(fp).replace("/Transparent/", "/Color/"), fp)
                Image.fromarray(rgb).save(OUT / f"ctx_{env}_{geom}_m{m}.png")
                L, T, side = crop_box(fp)
                rows.append(dict(env=env, geom=geom, metal=m, trial=info["trial"], frame=fp.name, left=L, top=T, side=side))
    pd.DataFrame(rows).to_csv(OUT / "boxes.csv", index=False)
    # one row through the metallic parameter (Sky, both shapes, head + hand block)
    for geom in ("Bumpy", "Smooth"):
        for m in (0, 0.25, 0.5, 0.75, 1):
            d, info = [(d, i) for d, i in tds if i["env"] == "Sky" and i["geom"] == geom and i["metal"] == m and i["modality"] == "both"][0]
            fr = frame_paths(d); fp = fr[len(fr) // 3]
            rgb, _ = context_crop(str(fp).replace("/Transparent/", "/Color/"), fp)
            Image.fromarray(rgb).save(OUT / f"levels_Sky_{geom}_m{m:.2f}.png")


def timecourse(trial=123, n=8):
    d, info = [x for x in trial_dirs(PID) if x[1]["trial"] == trial][0]
    fr = frame_paths(d)
    for f in OUT.glob("timecourse_*"):
        f.unlink()
    for k, i in enumerate(np.linspace(0, len(fr) - 1, n).round().astype(int)):
        fp = fr[i]
        t = (float(fp.stem) - float(fr[0].stem)) / 1000
        Image.open(str(fp).replace("/Transparent/", "/Color/")).convert("RGB").save(OUT / f"timecourse_{k}_{t:.2f}s.png")
        rgb, _ = context_crop(str(fp).replace("/Transparent/", "/Color/"), fp)
        Image.fromarray(rgb).save(OUT / f"timecourse_ctx_{k}_{t:.2f}s.png")
    print("timecourse", d.name, len(fr), "frames")


def pair(n=8):
    """Pick a trial ID at metallic 0.5 in the head+hand block where the own-video model
    (leave-one-stimulus-out scores, as in the disagreement test) separates a metal- and a
    glass-responder best, and save n frames of each video."""
    from common import MAIN_FEATURES
    S = pd.read_csv(RESULTS / "models" / f"{MAIN_FEATURES}_C0.1_scores.csv", dtype={"participant": str})
    T = pd.read_csv(ROOT / "data" / "trials.csv", dtype={"participant": str})
    S = S.merge(T[["participant", "trial", "folder", "metal", "env", "modality", "n_frames"]], on=["participant", "trial"])
    best = None
    for t, g in S[(S.metal == 0.5) & (S.modality == "both") & (S.env.isin(["Sky", "Studio"]))].groupby("trial"):
        m, gl = g[g.resp == 1], g[g.resp == 0]
        if len(m) < 3 or len(gl) < 3:
            continue
        m = m[(m.n_frames > 100) & (m.n_frames < 600)]; gl = gl[(gl.n_frames > 100) & (gl.n_frames < 600)]
        if len(m) == 0 or len(gl) == 0:
            continue
        a, b = m.loc[m.score_wt.idxmax()], gl.loc[gl.score_wt.idxmin()]
        gap = a.score_wt - b.score_wt
        if best is None or gap > best[0]:
            best = (gap, a, b)
    gap, a, b = best
    for f in OUT.glob("pair_*"):
        f.unlink()
    for lab, r in (("metal", a), ("glass", b)):
        d = IMGS / f"participant_{r.participant}" / "Transparent" / r.folder
        save_pair(frame_paths(d), f"pair_{lab}", n)
        print(lab, r.participant, r.folder, "score %.2f" % r.score_wt, "frames", r.n_frames)
    pd.DataFrame([dict(role="metal", participant=a.participant, folder=a.folder, score=a.score_wt, n_frames=a.n_frames),
                  dict(role="glass", participant=b.participant, folder=b.folder, score=b.score_wt, n_frames=b.n_frames)]).to_csv(OUT / "pair.csv", index=False)




def stats(trial=123):
    """For Figure A1: per-frame object statistics of every frame of one trial
    (object crops, as the models use them), the object crop and the
    object-plus-surround patch of one frame with its mask, and two consecutive
    object crops."""
    import extract_image_stats as E
    E.CROP = "object"
    d, info = [x for x in trial_dirs(PID) if x[1]["trial"] == trial][0]
    fr = frame_paths(d)
    crop_dir = IMGS / f"participant_{PID}" / "Cropped" / d.name
    rows, prevY, prevM = [], None, None
    for fp in fr:
        cp = crop_dir / fp.name
        if not cp.exists():
            continue
        rgb, obj = E.load_patch(d, cp)
        s, Y, region = E.frame_stats(rgb, obj)
        if s is None:
            continue
        s["t_s"] = (float(fp.stem) - float(fr[0].stem)) / 1000
        s["diff_energy"] = np.abs(Y - prevY)[region & prevM].mean() if prevY is not None else np.nan
        prevY, prevM = Y, region
        rows.append(s)
    pd.DataFrame(rows).to_csv(OUT / "timecourse_stats.csv", index=False)
    k = 6; i = int(np.linspace(0, len(fr) - 1, 8).round()[k])
    Image.open(crop_dir / fr[i].name).convert("RGB").save(OUT / "example_object.png")
    rgb, mask = context_crop(str(fr[i]).replace("/Transparent/", "/Color/"), fr[i])
    Image.fromarray(rgb).save(OUT / "example_patch.png")
    Image.fromarray((mask * 255).astype(np.uint8)).save(OUT / "example_patch_mask.png")
    j = len(fr) // 2
    for k2, jj in enumerate((j, j + 1)):
        Image.open(crop_dir / fr[jj].name).convert("RGB").save(OUT / f"consecutive_{k2}.png")
    for f in list(OUT.glob("timecourse_mask_*.png")):
        f.unlink()
    print("stats", len(rows), "frames")


def score_examples(n=3):
    """For Figure 2c: the lowest- and highest-scoring videos of observer PID under the
    main model (at least 100 frames), n object-plus-surround frames of each."""
    from common import MAIN_FEATURES
    S = pd.read_csv(RESULTS / "models" / f"{MAIN_FEATURES}_C0.1_scores.csv", dtype={"participant": str})
    T = pd.read_csv(ROOT / "data" / "trials.csv", dtype={"participant": str})
    S = S.merge(T[["participant", "trial", "folder", "metal", "env", "geom", "modality", "n_frames"]], on=["participant", "trial"])
    s = S[(S.participant == PID) & (S.n_frames >= 100)]
    for f in OUT.glob("score_*"):
        f.unlink()
    rows = []
    for role, r in (("low", s.loc[s.score.idxmin()]), ("high", s.loc[s.score.idxmax()])):
        d = IMGS / f"participant_{PID}" / "Transparent" / r.folder
        fr = frame_paths(d)
        for k, i in enumerate(np.linspace(0, len(fr) - 1, n).round().astype(int)):
            fp = fr[i]
            t = (float(fp.stem) - float(fr[0].stem)) / 1000
            rgb, _ = context_crop(str(fp).replace("/Transparent/", "/Color/"), fp)
            Image.fromarray(rgb).save(OUT / f"score_{role}_ctx_{k}_{t:.2f}s.png")
        rows.append(dict(role=role, trial=int(r.trial), score=r.score, resp=int(r.resp), env=r.env, geom=r.geom, metal=r.metal, modality=r.modality, n_frames=int(r.n_frames)))
        print(role, r.folder, "score %+.2f" % r.score, "resp", r.resp)
    pd.DataFrame(rows).to_csv(OUT / "score_examples.csv", index=False)


def trial_examples(trial=123, n=3):
    """For Figure 2d: among the 19 videos of one trial, the lowest- and highest-scoring
    under the main model (leave-one-stimulus-out scores, as in the disagreement test),
    n object-plus-surround frames of each."""
    from common import MAIN_FEATURES
    S = pd.read_csv(RESULTS / "models" / f"{MAIN_FEATURES}_C0.1_scores.csv", dtype={"participant": str})
    T = pd.read_csv(ROOT / "data" / "trials.csv", dtype={"participant": str})
    S = S.merge(T[["participant", "trial", "folder", "n_frames"]], on=["participant", "trial"])
    s = S[S.trial == trial]
    for f in OUT.glob("trial_*"):
        f.unlink()
    rows = []
    for role, r in (("low", s.loc[s.score_wt.idxmin()]), ("high", s.loc[s.score_wt.idxmax()])):
        d = IMGS / f"participant_{r.participant}" / "Transparent" / r.folder
        fr = frame_paths(d)
        for k, i in enumerate(np.linspace(0, len(fr) - 1, n).round().astype(int)):
            fp = fr[i]
            t = (float(fp.stem) - float(fr[0].stem)) / 1000
            rgb, _ = context_crop(str(fp).replace("/Transparent/", "/Color/"), fp)
            Image.fromarray(rgb).save(OUT / f"trial_{role}_ctx_{k}_{t:.2f}s.png")
        rows.append(dict(role=role, participant=r.participant, trial=trial, score=r.score_wt, resp=int(r.resp), n_frames=int(r.n_frames)))
        print(role, r.participant, r.folder, "score %+.2f" % r.score_wt, "resp", r.resp)
    pd.DataFrame(rows).to_csv(OUT / "trial_examples.csv", index=False)


if __name__ == "__main__":
    # python save_example_frames.py [stimuli] [timecourse] [stats] [pair] [score_examples] [trial_examples]
    what = sys.argv[1:] or ["stimuli", "timecourse"]
    for w in what:
        globals()[w]()
