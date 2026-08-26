#!/usr/bin/env bash
# One VM's share of stages 2+3. SHARD is 0..4, one per teammate.
set -euo pipefail
cd "$(dirname "$0")/.."

SHARD="${SHARD:?set SHARD=0..4}"
OF="${OF:-5}"
SCRATCH="${SCRATCH:-/dev/shm/mcv/variants}"
MODEL="${MODEL:-Qwen/Qwen3-VL-8B-Instruct}"

# HF_HOME is deliberately NOT defaulted here. This script used to set it to
# $HOME/hf_cache, which silently diverged from the plain `python -m ...` runs
# everyone actually does -- those use huggingface_hub's default
# ~/.cache/huggingface. The result was a second, empty cache: vLLM re-downloaded
# a 16GB checkpoint that was already on disk, filled the 90G root, and died
# three minutes in with "No space left on device". Inherit whatever the shell
# has, so this script and a manual run always agree.
export HF_HUB_ENABLE_HF_TRANSFER=1
HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"

echo "=== shard $SHARD/$OF on $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader

# Fail before the download, not partway through it. A judge checkpoint is
# ~16GB and the root disk is 90G shared with COCO, the base edits and any
# editor weights, so "is it already cached" is the question that matters.
echo "HF cache: $HF_CACHE"
MODEL_DIR="$HF_CACHE/hub/models--${MODEL//\//--}"
FREE_GB=$(df -BG --output=avail "$HOME" | tail -1 | tr -dc '0-9')
if [[ -d "$MODEL_DIR" ]]; then
  echo "  $MODEL is cached ($(du -sh "$MODEL_DIR" 2>/dev/null | cut -f1)); ${FREE_GB}G free"
elif (( FREE_GB < 25 )); then
  echo "ERROR: $MODEL is NOT cached and only ${FREE_GB}G is free." >&2
  echo "  A judge checkpoint needs ~20G. Free space first, or point HF_HOME at" >&2
  echo "  a cache that already has it:" >&2
  echo "    du -sh ~/.cache/huggingface/hub/models--* ~/hf_cache/hub/models--* 2>/dev/null" >&2
  exit 1
else
  echo "  $MODEL will be downloaded (~20G); ${FREE_GB}G free"
fi

# regenerate this shard's variants into RAM — never onto the 90G root disk
python -m src.stage2_corrupt --manifest out/manifest.parquet \
    --bases data/bases --out "$SCRATCH" --shard "$SHARD" --of "$OF"

python -m src.stage3_judge --manifest out/manifest.parquet \
    --bases data/bases --variants "$SCRATCH" --model "$MODEL" \
    --shard "$SHARD" --of "$OF" --out "out/scores_shard${SHARD}.parquet"

rm -rf "$SCRATCH"      # reclaim RAM; regenerating is cheap
echo "=== done: out/scores_shard${SHARD}.parquet ==="
