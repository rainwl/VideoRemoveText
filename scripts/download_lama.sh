#!/usr/bin/env bash
# The simple-lama-inpainting package downloads the LaMa weights on the
# first run automatically (~200MB). This helper just triggers that
# download ahead of time so the first real run is faster.
set -euo pipefail
python - <<'PY'
from simple_lama_inpainting import SimpleLama
print("Triggering LaMa weight download...")
SimpleLama()
print("Done.")
PY
