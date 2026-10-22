#!/bin/bash
# One-time setup, on the LOGIN node (it needs the internet; jobs run offline).
#
#   bash setup.sh            # after cloning, or: curl the file and run it
#
# Clones the repo, builds a Python 3.12 venv with the judge requirements, pulls
# the stage-1 tarball and both judge checkpoints into the default HF cache.
# Idempotent: re-running it skips whatever is already there.
set -euo pipefail

REPO="$HOME/mcv-spatial-audit"
BRANCH="${BRANCH:-athena-rerun}"
URL="https://github.com/nadavanolik/mcv-spatial-audit.git"
MODELS=(Qwen/Qwen3-VL-8B-Instruct OpenGVLab/InternVL3-8B)

echo "=== quota before (need ~50G: two 8B checkpoints ~34G, venv ~10G)"
quota -s || true

if [[ ! -d "$REPO/.git" ]]; then
    git clone -b "$BRANCH" "$URL" "$REPO"
fi
cd "$REPO"
git fetch origin
git checkout "$BRANCH"
git pull --ff-only

# numpy==1.26.4 and opencv-python-headless==4.11.0.86 have no 3.13 wheels, so
# the venv is 3.12 exactly. uv fetches that interpreter if the node lacks it.
command -v uv >/dev/null || { echo "uv not found on PATH" >&2; exit 1; }
[[ -x .venv/bin/python ]] || uv venv --seed --python 3.12 .venv
# shellcheck disable=SC1091
source .venv/bin/activate

# Installs requirements-judge.txt (vllm 0.11.0 brings its own torch, cu128,
# which the A100 driver 595 / CUDA 13.2 runs), prints the pins and the
# determinism hash. gate.job checks that hash again on a compute node.
bash scripts/setup.sh judge

# Stage-1 bases: 146MB, public. Never regenerated (invariant 4).
if [[ ! -f data/bases/bases.json ]]; then
    mkdir -p data
    TGZ=$(python -c "from huggingface_hub import hf_hub_download; print(hf_hub_download('mcv-spatial-audit/mcv-spatial-audit', 'bases.tar.gz', repo_type='dataset'))")
    tar xzf "$TGZ" -C data
fi
python -c "import json; b=json.load(open('data/bases/bases.json')); print('bases:', len(b))"

# Checkpoints into the default ~/.cache/huggingface (HF_HOME deliberately unset).
export HF_HUB_ENABLE_HF_TRANSFER=1
for M in "${MODELS[@]}"; do
    python -c "from huggingface_hub import snapshot_download as s; print(s('$M'))"
done

mkdir -p "$HOME/logs/mcv"
echo "=== quota after"
quota -s || true
echo "=== setup done. Next: bash jobs/athena/submit.sh"
