#!/usr/bin/env bash
# One command to bring a fresh machine to a verified state.
#
#   bash scripts/setup.sh judge     # stage 3 VM
#   bash scripts/setup.sh editor    # stage 1 VM
#   bash scripts/setup.sh coco      # whoever runs stage 0
#   bash scripts/setup.sh core      # CPU only (laptop, or a VM not yet assigned)
#
# Installs the right requirements file, prints the pinned versions the audit
# depends on, and runs the determinism check. The hash it prints at the end is
# the thing to compare against a teammate's.
set -euo pipefail
cd "$(dirname "$0")/.."

ROLE="${1:-core}"
case "$ROLE" in
  core)   REQ=requirements.txt ;;
  judge)  REQ=requirements-judge.txt ;;
  editor) REQ=requirements-editor.txt ;;
  coco)   REQ=requirements-coco.txt ;;
  *) echo "usage: bash scripts/setup.sh [core|judge|editor|coco]" >&2; exit 2 ;;
esac

echo "=== $ROLE setup on $(hostname) ==="

# Installing into the system python would need root we do not have, and would
# be shared state we cannot pin. Refuse rather than half-install.
if [[ -z "${VIRTUAL_ENV:-}${CONDA_PREFIX:-}" ]]; then
  echo "ERROR: no virtualenv or conda env is active." >&2
  echo "  python -m venv .venv && source .venv/bin/activate" >&2
  exit 1
fi
echo "env:    ${VIRTUAL_ENV:-$CONDA_PREFIX}"
echo "python: $(python -V 2>&1)  ($(command -v python))"

python -m pip install --upgrade pip
python -m pip install -r "$REQ"

echo
echo "=== pinned versions the corruption bytes depend on ==="
python - <<'PY'
import numpy, cv2, PIL
print("numpy   ", numpy.__version__)
print("cv2     ", cv2.__version__)
print("pillow  ", PIL.__version__)
PY

# One opencv, and it must be the headless build. A stray opencv-python or a
# conda opencv alongside it is a different library and can produce different
# corruption bytes, which silently breaks cross-VM comparability.
n_cv=$(python -m pip list 2>/dev/null | grep -ci '^opencv' || true)
if [[ "$n_cv" != "1" ]]; then
  echo "WARNING: expected exactly one opencv package, found $n_cv:" >&2
  python -m pip list 2>/dev/null | grep -i '^opencv' >&2 || true
  echo "  Uninstall all of them and reinstall opencv-python-headless alone." >&2
fi

echo
bash scripts/verify_determinism.sh

echo
echo "=== $ROLE ready. Post the CROSS-VM FIXTURE HASH above; all five must match. ==="
