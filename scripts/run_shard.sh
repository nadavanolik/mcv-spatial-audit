#!/usr/bin/env bash
# One VM's share of stages 2+3. SHARD is 0..4, one per teammate.
set -euo pipefail
cd "$(dirname "$0")/.."

SHARD="${SHARD:?set SHARD=0..4}"
OF="${OF:-5}"
SCRATCH="${SCRATCH:-/dev/shm/mcv/variants}"
MODEL="${MODEL:-Qwen/Qwen3-VL-8B-Instruct}"

export HF_HOME="${HF_HOME:-$HOME/hf_cache}"
export HF_HUB_ENABLE_HF_TRANSFER=1

echo "=== shard $SHARD/$OF on $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader

# regenerate this shard's variants into RAM — never onto the 90G root disk
python -m src.stage2_corrupt --manifest out/manifest.parquet \
    --bases data/bases --out "$SCRATCH" --shard "$SHARD" --of "$OF"

python -m src.stage3_judge --manifest out/manifest.parquet \
    --bases data/bases --variants "$SCRATCH" --model "$MODEL" \
    --shard "$SHARD" --of "$OF" --out "out/scores_shard${SHARD}.parquet"

rm -rf "$SCRATCH"      # reclaim RAM; regenerating is cheap
echo "=== done: out/scores_shard${SHARD}.parquet ==="
