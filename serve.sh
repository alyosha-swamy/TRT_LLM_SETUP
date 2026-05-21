#!/usr/bin/env bash
# =============================================================================
# Launch trtllm-serve against the AFMoE Trinity-Mini checkpoint.
# Assumes setup.sh has been run successfully.
# =============================================================================

set -euo pipefail

ROOT="${ROOT:-/alloc/trtllm-clean}"
MODEL_PATH="${MODEL_PATH:-arcee-ai/Trinity-Mini}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8765}"
SERVED_NAME="${SERVED_NAME:-trinity-mini}"
LOG="${LOG:-$ROOT/serve.log}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"

# shellcheck source=/dev/null
source "$ROOT/venv/bin/activate"

export CUDA_VISIBLE_DEVICES="$GPU"
export HF_HOME="${HF_HOME:-/alloc/hf-cache}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-/alloc/hf-cache/hub}"

echo "[serve] model:  $MODEL_PATH"
echo "[serve] gpu:    $GPU"
echo "[serve] listen: http://$HOST:$PORT  (served name: $SERVED_NAME)"
echo "[serve] logs:   $LOG"

exec trtllm-serve serve "$MODEL_PATH" \
  --host "$HOST" --port "$PORT" \
  --backend pytorch \
  --max_batch_size 8 \
  --max_num_tokens 4096 \
  --max_seq_len 4096 \
  --trust_remote_code \
  --served_model_name "$SERVED_NAME" \
  &> "$LOG"
