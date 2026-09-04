#!/usr/bin/env bash
# Build data/bases from COCO, from nothing, on a machine that has neither the
# annotations nor the images.
#
# This exists so base specs never have to be TRANSFERRED. Stage 0 is a pure
# function of (annotations file, config.yaml, --seed): same inputs, same 150
# photographs, same regions, same instructions, byte-identical under the pinned
# numpy / opencv / Pillow. Regenerating is ~30MB of download and a few minutes,
# against ~120MB of images in git history forever -- and it exercises the
# reproducibility claim instead of asserting it.
#
# Usage:
#     bash scripts/stage0.sh              # 150 bases from train2017
#     N=5 bash scripts/stage0.sh          # smaller, for a smoke run
#     FORCE=1 bash scripts/stage0.sh      # overwrite an existing data/bases
#
# Needs pycocotools:  bash scripts/setup.sh coco
set -euo pipefail
cd "$(dirname "$0")/.."

N="${N:-150}"
SPLIT="${SPLIT:-train2017}"
SEED="${SEED:-0}"
ANN_DIR="${ANN_DIR:-data/coco/annotations}"
IMG_DIR="${IMG_DIR:-data/coco/$SPLIT}"
OUT="${OUT:-data/bases}"
ANN="$ANN_DIR/instances_$SPLIT.json"
ZIP_URL="http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

echo "=== stage 0: $N bases from $SPLIT on $(hostname) ==="

# --- guards, all of them before any download ------------------------------

# pycocotools is the one dependency stage 0 has that no other stage needs, and
# it is deliberately NOT in requirements.txt. Fail here rather than after
# fetching 700MB of annotations.
python -c "import pycocotools" 2>/dev/null || {
  echo "ERROR: pycocotools is missing." >&2
  echo "  bash scripts/setup.sh coco     # or: pip install -r requirements-coco.txt" >&2
  exit 1
}

# Re-running on top of an existing data/bases is how you end up editing one set
# of instructions and judging another. Any change to the selection filter or
# the instruction templates restales every base, so the safe move is always to
# delete and regenerate, never to overwrite in place.
if [[ -d "$OUT" ]] && [[ -n "$(ls -A "$OUT" 2>/dev/null)" ]]; then
  if [[ -z "${FORCE:-}" ]]; then
    echo "ERROR: $OUT already exists and is not empty." >&2
    echo "  Stage 0 output is not merged in place. Delete it first:" >&2
    echo "    rm -rf $OUT out /dev/shm/mcv" >&2
    echo "  (out/ goes too: manifest.parquet embeds the instruction strings," >&2
    echo "   and any scores were judged against the old edits.)" >&2
    echo "  Then re-run, or set FORCE=1 to do it automatically." >&2
    exit 1
  fi
  echo "FORCE set; removing $OUT out /dev/shm/mcv"
  rm -rf "$OUT" out /dev/shm/mcv
fi

# The annotations JSON is ~450MB unpacked and the zip another 241MB alongside
# it. Cheap to check, annoying to discover half way through.
FREE_GB=$(df -BG --output=avail . | tail -1 | tr -dc '0-9')
if [[ ! -f "$ANN" ]] && (( FREE_GB < 3 )); then
  echo "ERROR: annotations need ~1G unpacked and only ${FREE_GB}G is free." >&2
  exit 1
fi

# --- 1. annotations (NOT the image split) ---------------------------------

# 241MB, and it carries train2017 and val2017 both. The 18GB image split is
# never downloaded: this filter keeps about 1 image in 100, so fetching the
# split would be 700x more download than the job needs and would not fit next
# to FLUX on a 90G root.
mkdir -p "$ANN_DIR"
if [[ -f "$ANN" ]]; then
  echo "[1/3] $ANN already present"
else
  echo "[1/3] downloading annotations (241MB zip)"
  ZIP="$ANN_DIR/annotations_trainval2017.zip"
  wget -q --show-progress -O "$ZIP" "$ZIP_URL"
  # unzip is not guaranteed on a VM with no sudo; python's zipfile always is.
  if command -v unzip >/dev/null; then
    unzip -o -j "$ZIP" "annotations/instances_$SPLIT.json" -d "$ANN_DIR"
  else
    python -c "import zipfile,shutil,sys; z=zipfile.ZipFile(sys.argv[1]); \
src=z.open('annotations/instances_'+sys.argv[2]+'.json'); \
dst=open(sys.argv[3],'wb'); shutil.copyfileobj(src,dst)" "$ZIP" "$SPLIT" "$ANN"
  fi
  rm -f "$ZIP"
fi

# --- 2. only the images that actually qualify -----------------------------

# --list-urls prints one URL per line on stdout and every other word on stderr,
# so this pipes straight into wget. It re-runs the same selection the real run
# will do, with the same seed, so the two agree on which images are needed.
echo "[2/3] resolving which images qualify"
mkdir -p "$IMG_DIR"
URLS=$(mktemp)
MISSING=$(mktemp)
# Both created before the trap: `set -u` makes a trap referencing an unset
# variable fail while the shell is already exiting, which hides the real error.
trap 'rm -f "$URLS" "$MISSING"' EXIT
python -m src.stage0_coco --coco "$ANN" --n "$N" --seed "$SEED" --list-urls > "$URLS"

WANT=$(wc -l < "$URLS")
if (( WANT < N )); then
  echo "WARNING: only $WANT images qualify, asked for $N." >&2
  echo "  Run with --survey for the breakdown before trusting the result." >&2
fi

# Fetch only what is not already on disk, so a re-run after an interrupted
# download costs nothing.
comm -13 <(ls -A "$IMG_DIR" | sort) <(sed 's#.*/##' "$URLS" | sort) > "$MISSING" || true
NEED=$(wc -l < "$MISSING")
if (( NEED == 0 )); then
  echo "      all $WANT images already present"
else
  # COCO photographs average ~160KB, so 150 of them is ~25MB, not 18GB.
  echo "      fetching $NEED of $WANT images (~$((NEED / 6))MB)"
  grep -F -f "$MISSING" "$URLS" | wget -q --show-progress -P "$IMG_DIR" -i -
fi

# --- 3. select, write masks, write bases.json ------------------------------

echo "[3/3] selecting bases"
python -m src.stage0_coco --coco "$ANN" --images "$IMG_DIR" \
       --out "$OUT" --n "$N" --seed "$SEED"

# --- report ---------------------------------------------------------------

BASES=$(find "$OUT" -mindepth 1 -maxdepth 1 -type d | wc -l)
REGIONS=$(find "$OUT" -name 'r*.png' | wc -l)
echo
echo "=== done: $BASES bases, $REGIONS regions in $OUT ==="
echo "Expected for N=150 on train2017: 150 bases, 476 regions, 3.17 regions/base."
echo "A different figure is not automatically wrong -- but it means your"
echo "selection differs from everyone else's, so check config.yaml's selection"
echo "block and your numpy/opencv/Pillow pins before running stage 1."
echo
echo "Next: build the manifest and compare its hash with the other VMs."
echo "  python -m src.build_manifest --profile main --out out/manifest.parquet"
