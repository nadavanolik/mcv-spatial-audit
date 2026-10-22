#!/bin/bash
# Shared environment for every Athena job. Sourced, never executed.
#
# SLURM does not source ~/.bashrc for batch jobs, so everything a job needs is
# exported here explicitly.

REPO="$HOME/mcv-spatial-audit"
PY="$REPO/.venv/bin/python"

# The venv first on PATH, as `source activate` would. scripts/*.sh call bare
# `python`, and without this a batch job gets the system interpreter, which has
# none of the pinned packages (gate 268712 died on "No module named 'cv2'").
export VIRTUAL_ENV="$REPO/.venv"
export PATH="$VIRTUAL_ENV/bin:$PATH"
GRID="$REPO/jobs/athena/grid.tsv"

# One results tree per submission. submit.sh exports RUN to every job, so the
# four jobs of one chain always write into the same directory.
RUN="${RUN:?RUN is unset; submit with jobs/athena/submit.sh}"
OUT="$REPO/results/$RUN"

# Values the gate refuses to run without. They are the cross-VM fixture hash
# four A10 VMs printed, and the manifest hashes in results/*/REPRODUCE.md. A
# mismatch means this machine would render different bytes or a different
# design matrix, and nothing it judges is comparable to the committed runs.
FIXTURE_HASH="776feeddd281fa726195bf504c7b19c8"
MANIFEST_SHA_main="1d862ad6ce725e26"
MANIFEST_SHA_nuisance150="8f5c7b61541fda69"

# Weights are downloaded once on the login node by setup.sh. Offline mode turns
# a cache miss into an error instead of a silent re-download, and pins every
# cell of the grid to the same snapshot.
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

# Variants are regenerated per job into node-local NVMe, never onto /home
# (invariant 1: regenerate, don't transfer). $TMPDIR is wiped when the job ends.
SCRATCH="${TMPDIR:-/tmp/$USER/${SLURM_JOB_ID:-manual}}/mcv"

# grid_cell <index>  -- 0-based, comments and blank lines skipped.
grid_cell() {
    grep -vE '^\s*(#|$)' "$GRID" | sed -n "$(($1 + 1))p"
}

# Nothing in here may fail the job: nvidia-smi exits non-zero on a CPU-only
# allocation, and under `set -e` that would kill the job mid-banner.
banner() {
    echo "=== ${SLURM_JOB_NAME:-manual}  job=${SLURM_JOB_ID:-}  task=${SLURM_ARRAY_TASK_ID:-}  node=$(hostname)  $(date -Is)"
    echo "=== run=$RUN  commit=$(git -C "$REPO" rev-parse --short HEAD)"
    echo "=== $*"
    echo "=== python: $($PY --version 2>&1)"
    if command -v nvidia-smi >/dev/null; then
        nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version --format=csv,noheader \
            || echo "(nvidia-smi found no devices)"
    fi
    echo
}
