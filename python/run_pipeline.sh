#!/bin/sh
# The whole Python pipeline, in order.  Stages 1 and 2 need the first-person
# frames (data/imgs, data/static; not distributed): skip them to recompute the
# model outputs from the shipped per-trial statistics (results/features/).
#   sh run_pipeline.sh            # everything
#   sh run_pipeline.sh models     # stage 3 onwards
set -e
cd "$(dirname "$0")"
PY=${PY:-python}
STAGE=${1:-all}

if [ "$STAGE" = "all" ]; then
  # 1. the trial table and the per-trial statistics
  $PY build_trial_table.py
  $PY extract_image_stats.py --crop object
  $PY extract_relational_stats.py          # every frame, then the summaries over the trial
  $PY extract_flow_cues.py                 # every frame pair, then the summaries over the trial
  $PY extract_time_windows.py              # a 2-s window slid back from the response in steps of 1 s
  # 2. the example frames of the figures (the pair and the trial examples need the model scores: see below)
  $PY save_example_frames.py stimuli timecourse stats
fi

# 3. the models: the main model, the 100 statistics without motion, each family alone
$PY behaviour_analysis.py
for fs in object_relational_flow object_relational image_stats_object relational flow; do
  $PY evaluate.py $fs
done
$PY regularization_grid.py
$PY pca_sweep.py
$PY static_block.py                        # needs data/static for the features; the shipped results/features/static/ is used when present
# 4. the disagreement test in detail, single statistics, observer-level tests
$PY within_trial_breakdown.py
$PY within_trial_by_similarity.py
$PY within_trial_by_difference.py
$PY within_trial_features.py
$PY sliding_window.py                      # the disagreement test on a 2-s window slid back from the response
$PY single_statistics.py
$PY observer_within_trial.py
$PY collect_results.py
$PY save_model_illustration.py
if [ "$STAGE" = "all" ]; then $PY save_example_frames.py pair score_examples trial_examples; fi
# 5. every number quoted in the paper
$PY numbers_for_text.py
echo PIPELINE DONE
