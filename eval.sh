#!/usr/bin/env bash
# =============================================================================
# Run openbench evals against a running trtllm-serve endpoint.
# Usage:  ./eval.sh [benchmarks...]
# Default benchmarks: gsm8k
# =============================================================================

set -euo pipefail

ROOT="${ROOT:-/alloc/trtllm-clean}"
BASE_URL="${BASE_URL:-http://127.0.0.1:8765/v1}"
SERVED_NAME="${SERVED_NAME:-trinity-mini}"
MAX_CONNECTIONS="${MAX_CONNECTIONS:-8}"
MAX_TOKENS="${MAX_TOKENS:-3500}"      # leaves room for thinking budget
TEMPERATURE="${TEMPERATURE:-0.0}"
LIMIT="${LIMIT:-200}"
OUT_DIR="${OUT_DIR:-$ROOT/results}"
OPENBENCH="${OPENBENCH:-/alloc/vllm-env/bin/openbench}"

mkdir -p "$OUT_DIR/logs"

# Pick up HF token / inference keys without leaking them into the repo
if [[ -f /alloc/evals/env.sh ]]; then
  # shellcheck disable=SC1091
  source /alloc/evals/env.sh
fi
export OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}"

BENCHMARKS=("${@:-gsm8k}")

echo "[eval] base_url:   $BASE_URL"
echo "[eval] model:      openai/$SERVED_NAME"
echo "[eval] limit:      $LIMIT"
echo "[eval] benchmarks: ${BENCHMARKS[*]}"

# Wait until the endpoint is alive (max 5 min)
for _ in $(seq 1 60); do
  if curl -sf "$BASE_URL/models" >/dev/null 2>&1; then break; fi
  sleep 5
done
curl -sf "$BASE_URL/models" >/dev/null || { echo "[eval] server not reachable"; exit 1; }

"$OPENBENCH" eval "${BENCHMARKS[@]}" \
  --model "openai/$SERVED_NAME" \
  --model-base-url "$BASE_URL" \
  --max-connections "$MAX_CONNECTIONS" \
  --max-tokens "$MAX_TOKENS" \
  --temperature "$TEMPERATURE" \
  --limit "$LIMIT" \
  --log-dir "$OUT_DIR/logs" \
  2>&1 | tee "$OUT_DIR/eval.log"
