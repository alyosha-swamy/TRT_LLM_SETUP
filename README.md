# AFMoE on TensorRT-LLM — clean environment setup

This repository contains a reproducible recipe for serving and evaluating the
**AFMoE (Arcee Foundation MoE)** architecture under TensorRT-LLM, starting
from a clean machine that has GPUs, `uv`, and `git`. AFMoE lives on the
[`afmoe-trinity-support`](https://github.com/alyosha-swamy/TensorRT-LLM/tree/afmoe-trinity-support)
branch of TensorRT-LLM (proposed upstream in
[NVIDIA/TensorRT-LLM#13148](https://github.com/NVIDIA/TensorRT-LLM/pull/13148));
this repo wires the branch into a working install + serve + eval pipeline
without requiring a source build.

## What this gives you

| Script       | Purpose                                                          |
| ------------ | ---------------------------------------------------------------- |
| `setup.sh`   | Build a fresh venv, install TRT-LLM rc14 wheel, overlay AFMoE.   |
| `serve.sh`   | Launch `trtllm-serve` against the Trinity-Mini AFMoE checkpoint. |
| `eval.sh`    | Run `openbench` evals against the running server.                |

All artifacts go under `$ROOT` (default `/alloc/trtllm-clean`).

## Background — why is the layout the way it is?

The AFMoE branch (`alyosha-swamy/TensorRT-LLM @ afmoe-trinity-support`) is
based on TRT-LLM **rc15** source. NVIDIA has not (yet) published an rc15
pre-built wheel; the most recent public wheel is **rc14**. We therefore:

1. Install the public **rc14 wheel** for the heavy C++/CUDA binaries
   (libtensorrt_llm.so, plugins, bindings, …).
2. Overlay only the two new AFMoE Python files from the branch:
   * `tensorrt_llm/_torch/models/modeling_afmoe.py`
   * `tensorrt_llm/_torch/models/checkpoints/hf/afmoe_weight_mapper.py`
3. Append one `import` line to each of two `__init__.py` files so the new
   modules are loaded at package import time. AFMoE registers itself via
   `@register_auto_model("AfmoeForCausalLM")` and
   `@register_mapper("HF", "AfmoeForCausalLM")`, so once the modules are
   imported, `LLM`, `trtllm-serve`, `trtllm-bench`, etc. see the architecture
   automatically.

This works because the AFMoE branch only uses public APIs that already exist
in rc14. The rest of the rc15 changes on the branch are not needed at
runtime.

When a matching `tensorrt-llm==1.3.0rc15` wheel ships, change
`TRTLLM_VERSION` and re-run `setup.sh`.

## Prerequisites

* Linux + NVIDIA GPU with a recent CUDA driver (tested on H100, driver
  compatible with CUDA 13).
* `uv` (`curl -LsSf https://astral.sh/uv/install.sh | sh`).
* `git`.
* ~30 GB free disk in `$ROOT` (wheel + libs dominate).
* Internet access to `pypi.org` and `pypi.nvidia.com`.

## Quick start

```bash
# 1. Build the environment (one-time, ~5–10 min depending on bandwidth).
./setup.sh

# 2. (Optional) pre-download the Trinity-Mini checkpoint into your HF cache.
#    Uses HF_TOKEN from your shell. If you skip this, serve.sh will let
#    HuggingFace pull weights on the first run.
hf download arcee-ai/Trinity-Mini

# 3. Launch the server (foreground; usually run with nohup/tmux).
./serve.sh

# 4. From another shell, run the evals.
LIMIT=50 ./eval.sh gsm8k
```

## Configuration knobs

All scripts respect environment variables:

| Variable             | Default                                  | Meaning                                                   |
| -------------------- | ---------------------------------------- | --------------------------------------------------------- |
| `ROOT`               | `/alloc/trtllm-clean`                    | Where the venv and source checkout live.                  |
| `REPO_URL`           | `https://github.com/alyosha-swamy/TensorRT-LLM.git` | AFMoE branch fork.                                       |
| `BRANCH`             | `afmoe-trinity-support`                  | Branch to overlay.                                        |
| `TRTLLM_VERSION`     | `1.3.0rc14`                              | Pre-built wheel version to install.                       |
| `PYTHON_VERSION`     | `3.10`                                   | Python used for the venv.                                 |
| `MODEL_PATH`         | `arcee-ai/Trinity-Mini`                  | Local path or HF id of the AFMoE checkpoint.              |
| `HOST` / `PORT`      | `127.0.0.1` / `8765`                     | trtllm-serve bind address.                                |
| `SERVED_NAME`        | `trinity-mini`                           | OpenAI-API model id exposed by the server.                |
| `CUDA_VISIBLE_DEVICES` | `0`                                    | Which GPU(s) the server uses.                             |
| `BASE_URL`           | `http://127.0.0.1:8765/v1`               | Server URL the eval client talks to.                      |
| `MAX_TOKENS`         | `3500`                                   | Per-sample generation cap (Trinity-Mini is a thinking model). |
| `LIMIT`              | `200`                                    | Per-benchmark sample cap.                                 |
| `OUT_DIR`            | `$ROOT/results`                          | Where eval logs are written.                              |

## Step-by-step explanation of `setup.sh`

1. **Create venv.** `uv venv --python 3.10 venv` gives a clean isolated
   Python.
2. **Install build deps.** Many wheels still need `setuptools`/`wheel` even
   when installed via uv, so we pre-seed them.
3. **Install TRT-LLM rc14 wheel.** Two flags matter:
   * `--extra-index-url https://pypi.nvidia.com` — TRT-LLM, cuda-python,
     cutlass, etc. only live on NVIDIA's index.
   * `--index-strategy unsafe-best-match` — without this, `cuda-python>=13`
     resolves against PyPI only and fails (no public 13.x release there).
4. **Clone AFMoE branch.** Shallow clone (`--depth 50`) is enough; we only
   need the AFMoE files plus a few merge-base commits for diagnostics.
5. **Overlay the two AFMoE files** into the installed `tensorrt_llm`
   package. The package location is discovered with
   `importlib.util.find_spec("tensorrt_llm")` rather than `import
   tensorrt_llm`, because the package prints a banner and a couple of
   harmless `🚨 Config not found for parakeet…` warnings to stdout the
   first time it is imported. Those would otherwise get captured into the
   `SITE` variable and break the path math.
6. **Append one import line** to each of two `__init__.py` files. The
   `append_once` helper is idempotent — re-running `setup.sh` will not
   stack duplicate lines.
7. **Sanity-check** by importing `AfmoeForCausalLM` and
   `AfmoeHfWeightMapper`. If this print succeeds, the rest of the toolchain
   (`trtllm-serve`, `trtllm-bench`, `trtllm-eval`, plain `LLM(...)`) will
   discover the architecture.

## Step-by-step explanation of `serve.sh`

* Activates the venv from `$ROOT`.
* Restricts visible GPUs (default `0`) and points HF caches to the shared
  `/alloc/hf-cache` so model downloads are cached across runs.
* Launches `trtllm-serve serve` with conservative limits suitable for a
  single H100:
  * `--max_batch_size 8` — Trinity-Mini is ~7B active params with thinking,
    so 8 concurrent requests is a safe sweet spot.
  * `--max_num_tokens 4096` / `--max_seq_len 4096` — leaves enough budget
    for long chain-of-thought completions while keeping KV-cache modest.
  * `--trust_remote_code` — required for the AFMoE HF config class.
  * `--served_model_name trinity-mini` — fixes the OpenAI-API model id so
    eval clients can hard-code it.

## Step-by-step explanation of `eval.sh`

* Sources `/alloc/evals/env.sh` if present to pick up `HF_TOKEN` etc.
* Polls `${BASE_URL}/models` for up to 5 minutes so it can be started in
  parallel with `serve.sh`.
* Runs `openbench eval` with `max-tokens=3500`. This is **important** for
  Trinity-Mini: with the default `max-tokens=512` the model hits the limit
  inside its `<think>...</think>` block on nearly every sample and never
  emits a final answer, which crashes accuracy. With the larger budget the
  model can finish its CoT and the scorer parses the answer correctly.
* Writes logs to `$OUT_DIR` (default `$ROOT/results`).

## Quick verification commands

```bash
# 1) Architecture is registered.
source /alloc/trtllm-clean/venv/bin/activate
python -c "from tensorrt_llm._torch.models import AfmoeForCausalLM; print(AfmoeForCausalLM)"

# 2) Server responds.
curl -s http://127.0.0.1:8765/v1/models | jq .

# 3) Smoke completion.
curl -s http://127.0.0.1:8765/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"trinity-mini","prompt":"Hello","max_tokens":16}' | jq .
```

## Verified output (May 2026, single H100)

End-to-end run of `setup.sh -> serve.sh -> eval.sh gsm8k` on a clean H100
node, with `LIMIT=30 MAX_TOKENS=3500`:

```
[setup] Installed tensorrt_llm package lives at:
        /alloc/trtllm-clean/venv/lib/python3.10/site-packages/tensorrt_llm
tensorrt_llm        : 1.3.0rc14
AfmoeForCausalLM    -> tensorrt_llm._torch.models.modeling_afmoe
AfmoeHfWeightMapper -> tensorrt_llm._torch.models.checkpoints.hf.afmoe_weight_mapper
AFMoE registered OK

gsm8k (30 samples): openai/trinity-mini
  total time            : 0:01:00
  tokens                : 48,861  (in 3,203 / out 45,658)
  avg sample duration   : 14.21 s
  p95 sample duration   : 31.56 s
  grade_school_math_scorer
    accuracy            : 0.933
    stderr              : 0.046
```

For reference, the same model evaluated with `MAX_TOKENS=512` (i.e. with the
thinking budget chopped off) collapses to ≈ 0.22 accuracy — so always size
`MAX_TOKENS` generously for reasoning models, even if your prompts are short.

## Troubleshooting

* **`No solution found ... cuda-python>=13`** — you forgot
  `--index-strategy unsafe-best-match`. `setup.sh` passes it; if you run
  `uv pip install` manually, include it too.
* **`NotImplementedError: AfmoeForCausalLM is not supported in TRT-LLM yet`**
  — the overlay step in `setup.sh` did not run against the binary you are
  using. Confirm `which trtllm-serve` resolves inside `$ROOT/venv/bin/` and
  that the two `# --- AFMoE overlay ---` lines exist in:
  * `$SITE/_torch/models/__init__.py`
  * `$SITE/_torch/models/checkpoints/__init__.py`
* **`ImportError: cannot import name 'fp8_fp4_mqa_logits'`** — you installed
  rc15 source against rc14 binaries without doing the overlay correctly.
  Re-run `setup.sh`; it pins the source overlay to AFMoE-only files so the
  rest of the package stays at rc14.
* **GSM8K accuracy near zero** — almost always `max-tokens` is too small for
  the thinking model. Use `MAX_TOKENS=3500` (default) or higher.
