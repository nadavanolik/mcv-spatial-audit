#!/bin/bash
# Submit the whole chain. Run on the login node, after setup.sh.
#
#   bash jobs/athena/submit.sh                 # RUN=athena_<today>
#   RUN=athena_2026-10-22b bash jobs/athena/submit.sh
#
#   gate (GPU, ~45 min) --afterok--> judge array (29 GPU cells, ~3 h)  --afterany--+
#                       --afterok--> checks (CPU, ~30 min)             --afterok---+--> analyze (CPU, ~30 min)
#
# A failed gate cancels everything behind it (--kill-on-invalid-dep), so no
# GPU hour is spent on a broken environment.
set -euo pipefail
cd "$(dirname "$0")"

export RUN="${RUN:-athena_$(date +%F)}"
mkdir -p "$HOME/logs/mcv"
SB=(sbatch --parsable --kill-on-invalid-dep=yes --export=ALL,RUN="$RUN")

GATE=$("${SB[@]}" gate.job)
JUDGE=$("${SB[@]}" --dependency=afterok:"$GATE" judge.job)
CHECKS=$("${SB[@]}" --dependency=afterok:"$GATE" checks.job)
ANALYZE=$("${SB[@]}" --dependency=afterany:"$JUDGE",afterok:"$CHECKS" analyze.job)

echo "RUN=$RUN"
echo "gate    $GATE"
echo "judge   $JUDGE   (array, 29 cells)"
echo "checks  $CHECKS"
echo "analyze $ANALYZE"
echo "results: ~/mcv-spatial-audit/results/$RUN   logs: ~/logs/mcv/"
echo "watch:   squeue -u \$USER -o '%.12i %.14j %.8T %.10M %R'"
