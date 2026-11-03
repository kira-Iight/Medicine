#!/bin/bash
set -e
unset CLUSTERBP_PRESET CLUSTERBP_TAG CLUSTERBP_RAW_DIR
for nb in clusterbp_v2_temporal clusterbp_v2_moe clusterbp_v2_sequence_models \
          clusterbp_v2_regions clusterbp_v2_additional clusterbp_v2_interpret \
          clusterbp_v2_leakage clusterbp_v2_first_events clusterbp_v2_features clusterbp_v2_shift_subgroups; do
  echo "=== $(date +%H:%M) start $nb"
  jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 "$nb.ipynb"
  echo "=== $(date +%H:%M) done  $nb"
done
echo "=== ALL DONE"
