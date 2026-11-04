"""Leave-one-observer-out and within-trial AUC of the main model over a grid
of the regularization constant (no permutations, no swaps), for the Appendix.
Writes results/models/regularization_grid.csv."""
import pandas as pd
from evaluate import evaluate
from common import RESULTS, MAIN_FEATURES

rows = []
for C in (0.001, 0.01, 0.1, 1.0, 10.0):
    res, _, _ = evaluate(MAIN_FEATURES, C=C, n_perm=0, n_swap=0)
    rows.append(dict(C=C, auc_lopo=res["auc_lopo"], within_trial_auc=res["within_trial_auc"]))
    print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(RESULTS / "models" / "regularization_grid.csv", index=False)
