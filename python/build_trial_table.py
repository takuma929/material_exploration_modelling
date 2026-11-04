"""Scan data/imgs and write data/trials.csv: one row per participant x trial."""
import pandas as pd
from common import IMGS, ROOT, participants, parse_trial_dir

rows = []
for pid in participants():
    for variant in ("Color", "Cropped", "Transparent"):
        vdir = IMGS / f"participant_{pid}" / variant
        if not vdir.is_dir():
            continue
        for d in sorted(vdir.iterdir()):
            info = parse_trial_dir(d.name)
            if info is None or not d.is_dir():
                continue
            frames = sorted(float(f.stem) for f in d.glob("*.png"))
            rows.append(dict(participant=pid, variant=variant, folder=d.name, **info,
                             n_frames=len(frames),
                             t_first_ms=frames[0] if frames else float("nan"),
                             t_last_ms=frames[-1] if frames else float("nan")))
df = pd.DataFrame(rows)
wide = df[df.variant == "Color"].drop(columns="variant").copy()
for v in ("Cropped", "Transparent"):
    n = df[df.variant == v].set_index(["participant", "trial"])["n_frames"]
    wide[f"n_frames_{v.lower()}"] = wide.set_index(["participant", "trial"]).index.map(n).values
wide = wide.sort_values(["participant", "trial"]).reset_index(drop=True)
wide.to_csv(ROOT / "data" / "trials.csv", index=False)
print(wide.groupby("participant").size().describe())
print(wide.head())
print("mismatch Color/Cropped frame counts:", int((wide.n_frames != wide.n_frames_cropped).sum()))
