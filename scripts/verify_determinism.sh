#!/usr/bin/env bash
# Run on TWO different VMs and compare the printed hash.
# If they differ, scores from different shards are not comparable and the
# audit is invalid. Fix before generating anything you intend to report.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "host: $(hostname)"
python -c "import numpy,cv2,PIL;print('numpy',numpy.__version__,'cv2',cv2.__version__,'pillow',PIL.__version__)"
python tests/test_determinism.py

python - <<'PY'
import hashlib, sys; sys.path.insert(0,'.')
import numpy as np
from src.corruptions import apply_corruption
from src.schema import seed_for, variant_id
rng=np.random.default_rng(0)
img=rng.integers(0,255,(256,256,3),dtype=np.uint8)
mask=np.zeros((256,256),np.uint8); mask[64:160,64:160]=255
h=hashlib.sha256()
for c in ["blur","saturate","noise","jpeg","remove"]:
    for s in (1,2,3):
        for a in ("full","half","quarter"):
            v=variant_id("fixture",0,c,s,a)
            h.update(apply_corruption(img,mask,c,s,a,seed_for(v)).tobytes())
print("CROSS-VM FIXTURE HASH:", h.hexdigest()[:32])
PY
