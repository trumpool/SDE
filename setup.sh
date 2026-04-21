#!/usr/bin/env bash
# One-shot bootstrap: clone → ``bash setup.sh`` → all tests green + BERT cache
# ready + a 10-step sanity training run. Works on macOS (system Python 3.9+)
# and Linux (python3.10+ recommended). Re-runs are idempotent: every step
# skips itself if its artifact already exists.
#
# Flags:
#   --quick      Skip the 20-min BERT encoding and the training sanity run.
#   --no-train   Encode, but skip the end-to-end training sanity run.
#   --cpu-train  Force ``--device cpu`` for the sanity run (default anyway on
#                Macs since MPS is unexpectedly slow on the unrolled
#                integrator; see OPEN_ISSUES §16).
#   --device X   Passed through to train_weibo.py (``cpu``/``mps``/``cuda``).
#   -h, --help   This help text.

set -euo pipefail

QUICK=0
SKIP_TRAIN=0
DEVICE="cpu"

for arg in "$@"; do
    case "$arg" in
        --quick)     QUICK=1 ;;
        --no-train)  SKIP_TRAIN=1 ;;
        --cpu-train) DEVICE="cpu" ;;
        --device=*)  DEVICE="${arg#--device=}" ;;
        --device)    shift; DEVICE="${1:-cpu}" ;;
        -h|--help)
            sed -n '2,18p' "$0"
            exit 0
            ;;
        *)
            echo "unknown flag: $arg" >&2
            exit 2
            ;;
    esac
done

# -------------- repo root --------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

step() { printf '\n\033[1;32m[setup] %s\033[0m\n' "$*"; }

# -------------- Python interpreter --------------
if ! command -v python3 >/dev/null 2>&1; then
    echo "ERROR: python3 not found in PATH" >&2
    exit 1
fi
PYBIN="$(command -v python3)"
step "using $PYBIN ($($PYBIN --version))"

# -------------- venv --------------
if [ ! -d .venv ]; then
    step "creating venv at .venv"
    "$PYBIN" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade --quiet pip

step "installing requirements"
pip install --quiet -r requirements.txt
pip install --quiet pytest

# Quick import sanity — fail fast with a readable error if a key dep broke.
python - <<'PY'
import torch, torchsde, scipy, pandas, gdown, transformers
print(f"[sanity] torch={torch.__version__}  mps={torch.backends.mps.is_available()}  cuda={torch.cuda.is_available()}")
print(f"[sanity] torchsde={torchsde.__version__}  transformers={transformers.__version__}")
PY

# -------------- tests --------------
step "running unit tests"
python -m pytest tests/ -q

# -------------- raw data --------------
mkdir -p data/raw data/encoded
if [ ! -f data/raw/2019-12.csv ]; then
    step "downloading 2019-12.csv from Drive (~60 MB)"
    python - <<'PY'
import gdown
gdown.download(
    "https://drive.google.com/uc?id=1dakfZtBG0itJTHc3_544t2sPHplTpqW_",
    "data/raw/2019-12.csv",
    quiet=False,
)
PY
else
    step "data/raw/2019-12.csv already present; skipping download"
fi

# -------------- BERT cache --------------
if [ "$QUICK" -eq 1 ]; then
    step "QUICK mode: skipping BERT encoding + training run"
    step "setup done"
    exit 0
fi

if [ ! -f data/encoded/2019-12_cls.pt ]; then
    step "encoding 2019-12 with RoBERTa-wwm-ext (~20 min on Apple MPS)"
    python scripts/encode_weibo.py --csv data/raw/2019-12.csv
else
    step "data/encoded/2019-12_cls.pt already present; skipping encoding"
fi

# -------------- training sanity run --------------
if [ "$SKIP_TRAIN" -eq 1 ]; then
    step "--no-train flag set; skipping end-to-end run"
else
    step "end-to-end training sanity run (10 steps, device=$DEVICE)"
    python scripts/train_weibo.py \
        --bert-cache data/encoded/2019-12_cls.pt \
        --max-seqs 30 --steps 10 \
        --beta 0.1 --device "$DEVICE"
fi

step "setup done — repo is now in the same state as commit $(git rev-parse --short HEAD)"
