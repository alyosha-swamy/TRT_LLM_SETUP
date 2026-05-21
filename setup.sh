#!/usr/bin/env bash
# =============================================================================
# AFMoE on TensorRT-LLM: clean from-scratch environment setup.
# =============================================================================
#
# This script builds a fresh, reproducible Python environment that can serve
# and evaluate the AFMoE (Arcee Foundation MoE) architecture inside
# TensorRT-LLM. The AFMoE branch is built on top of TRT-LLM rc15 source, but
# the latest publicly-published pre-built wheel is rc14. We therefore install
# the rc14 wheel for binaries and overlay AFMoE's two new Python files on top
# (the modules register themselves via @register_auto_model / @register_mapper
# decorators, so no other source edits are needed).
#
# Outputs (all relative to $ROOT):
#   venv/                          - clean Python 3.10 virtualenv
#   src/                           - AFMoE branch checkout (source of truth)
#   venv/.../site-packages/tensorrt_llm/_torch/models/modeling_afmoe.py
#   venv/.../site-packages/tensorrt_llm/_torch/models/checkpoints/hf/afmoe_weight_mapper.py
#
# Prereqs: NVIDIA GPU with CUDA, uv installed, git, ~30 GB disk.
# =============================================================================

set -euo pipefail

ROOT="${ROOT:-/alloc/trtllm-clean}"
REPO_URL="${REPO_URL:-https://github.com/alyosha-swamy/TensorRT-LLM.git}"
BRANCH="${BRANCH:-afmoe-trinity-support}"
TRTLLM_VERSION="${TRTLLM_VERSION:-1.3.0rc14}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"

log() { printf "\n\033[1;36m[setup]\033[0m %s\n" "$*"; }

log "Workspace: $ROOT"
mkdir -p "$ROOT"
cd "$ROOT"

# -----------------------------------------------------------------------------
# 1. Fresh virtualenv via uv
# -----------------------------------------------------------------------------
if [[ ! -d "$ROOT/venv" ]]; then
  log "Creating Python ${PYTHON_VERSION} venv with uv"
  uv venv --python "${PYTHON_VERSION}" "$ROOT/venv"
fi
# shellcheck source=/dev/null
source "$ROOT/venv/bin/activate"

# -----------------------------------------------------------------------------
# 2. Base build tooling
# -----------------------------------------------------------------------------
log "Installing pip / setuptools / wheel into the venv"
uv pip install --quiet pip setuptools wheel

# -----------------------------------------------------------------------------
# 3. TensorRT-LLM binaries (pre-built wheel)
#    Two non-obvious things to know:
#      a) tensorrt-llm rc14 pins cuda-python>=13, which forces a CUDA-13 build
#         of torch. The default PyPI torch wheel is built against CUDA 12.9
#         (its `cuda-bindings==12.9.4` dep makes it incompatible). We must
#         explicitly install torch==2.10.0+cu130 from the PyTorch CU130 index
#         BEFORE letting uv resolve tensorrt-llm.
#      b) Everything CUDA-related on the NVIDIA index needs
#         --index-strategy unsafe-best-match so uv prefers the NVIDIA index
#         over PyPI for matching package names (e.g. cuda-python, cutlass).
# -----------------------------------------------------------------------------
log "Installing torch==2.10.0+cu130 from PyTorch CU130 index"
uv pip install \
  --index-strategy unsafe-best-match \
  --extra-index-url https://download.pytorch.org/whl/cu130 \
  "torch==2.10.0+cu130" "torchvision" "torchaudio"

log "Installing tensorrt-llm==${TRTLLM_VERSION} (this is the heavy step ~5GB)"
uv pip install \
  --prerelease=allow \
  --index-strategy unsafe-best-match \
  --extra-index-url https://pypi.nvidia.com \
  --extra-index-url https://download.pytorch.org/whl/cu130 \
  "tensorrt-llm==${TRTLLM_VERSION}"

# -----------------------------------------------------------------------------
# 4. AFMoE source checkout
# -----------------------------------------------------------------------------
if [[ ! -d "$ROOT/src/.git" ]]; then
  log "Cloning AFMoE branch from $REPO_URL ($BRANCH)"
  git clone --branch "$BRANCH" --depth 50 "$REPO_URL" "$ROOT/src"
else
  log "Updating existing AFMoE checkout"
  git -C "$ROOT/src" fetch origin "$BRANCH"
  git -C "$ROOT/src" checkout "$BRANCH"
  git -C "$ROOT/src" reset --hard "origin/$BRANCH"
fi

# -----------------------------------------------------------------------------
# 5. Overlay AFMoE files on the installed package
#    AFMoE adds exactly two new Python modules; both self-register via
#    decorators when imported, so all we have to do is (a) drop them into
#    site-packages and (b) make sure they get imported on package load.
# -----------------------------------------------------------------------------
# NOTE: do not `import tensorrt_llm` here -- its __init__ prints a banner and
# warnings to stdout, which would poison the SITE capture. find_spec only
# resolves the path on disk without executing the package.
SITE="$(python -c 'import importlib.util, os; s = importlib.util.find_spec("tensorrt_llm"); print(os.path.dirname(s.origin))' 2>/dev/null)"
log "Installed tensorrt_llm package lives at: $SITE"
if [[ -z "$SITE" || ! -d "$SITE/_torch/models" ]]; then
  echo "[setup] could not locate installed tensorrt_llm package" >&2
  exit 1
fi

log "Copying AFMoE model + weight-mapper modules"
install -m 0644 \
  "$ROOT/src/tensorrt_llm/_torch/models/modeling_afmoe.py" \
  "$SITE/_torch/models/modeling_afmoe.py"
install -d "$SITE/_torch/models/checkpoints/hf"
install -m 0644 \
  "$ROOT/src/tensorrt_llm/_torch/models/checkpoints/hf/afmoe_weight_mapper.py" \
  "$SITE/_torch/models/checkpoints/hf/afmoe_weight_mapper.py"

append_once() {
  local file="$1" line="$2"
  if ! grep -qxF "$line" "$file"; then
    printf '\n# --- AFMoE overlay ---\n%s\n' "$line" >> "$file"
    log "Appended to $file:  $line"
  fi
}

append_once \
  "$SITE/_torch/models/__init__.py" \
  "from .modeling_afmoe import AfmoeForCausalLM  # noqa: F401"
append_once \
  "$SITE/_torch/models/checkpoints/__init__.py" \
  "from .hf.afmoe_weight_mapper import AfmoeHfWeightMapper  # noqa: F401"

# -----------------------------------------------------------------------------
# 6. Sanity check
# -----------------------------------------------------------------------------
log "Verifying AFMoE registration"
python - <<'PY'
import tensorrt_llm
from tensorrt_llm._torch.models import AfmoeForCausalLM
from tensorrt_llm._torch.models.checkpoints import AfmoeHfWeightMapper
print(f"tensorrt_llm  : {tensorrt_llm.__version__}")
print(f"AfmoeForCausalLM       -> {AfmoeForCausalLM.__module__}")
print(f"AfmoeHfWeightMapper    -> {AfmoeHfWeightMapper.__module__}")
print("AFMoE registered OK")
PY

log "Setup complete."
log "Activate with:  source $ROOT/venv/bin/activate"
