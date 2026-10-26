#!/usr/bin/env bash
# One machine's share of stages 2+3. SHARD is 0..4.
set -euo pipefail
cd "$(dirname "$0")/.."

SHARD="${SHARD:?set SHARD=0..4}"
OF="${OF:-5}"
SCRATCH="${SCRATCH:-/dev/shm/mcv/variants}"
MODEL="${MODEL:-Qwen/Qwen3-VL-8B-Instruct}"

# HF_HOME is inherited, not set, so this script and a manual run share one
# model cache.
export HF_HUB_ENABLE_HF_TRANSFER=1
HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"

echo "=== shard $SHARD/$OF on $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader

# Fail before downloading if the model is not cached and the disk is short.
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

# Render this shard's variants into RAM-backed scratch.
python -m src.stage2_corrupt --manifest out/manifest.parquet \
    --bases data/bases --out "$SCRATCH" --shard "$SHARD" --of "$OF"

# Greedy decoding, passed explicitly and not overridable: every shard must be
# decoded the same way.
python -m src.stage3_judge --manifest out/manifest.parquet \
    --bases data/bases --variants "$SCRATCH" --model "$MODEL" \
    --shard "$SHARD" --of "$OF" --temperature 0 --n-samples 1 \
    --out "out/scores_shard${SHARD}.parquet"

# Free the scratch space (re-rendering is cheap). KEEP_SCRATCH=1 keeps it.
if [[ -n "${KEEP_SCRATCH:-}" ]]; then
  echo "KEEP_SCRATCH set; leaving $SCRATCH in place"
else
  rm -rf "$SCRATCH"
fi
echo "=== done: out/scores_shard${SHARD}.parquet ==="
