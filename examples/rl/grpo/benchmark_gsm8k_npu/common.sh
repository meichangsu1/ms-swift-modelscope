#!/usr/bin/env bash
# Shared settings: two training NPUs plus two rollout NPUs. Source from a launcher.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SWIFT_DIR="${SWIFT_DIR:-$(cd -- "$SCRIPT_DIR/../../../.." && pwd)}"
TWINKLE_DIR="${TWINKLE_DIR:-$(dirname -- "$SWIFT_DIR")/twinkle}"
VERL_DIR="${VERL_DIR:-$(dirname -- "$SWIFT_DIR")/verl}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL_PATH="${MODEL_PATH:-/nas/disk1/Qwen3-4B}"
DATA_DIR="${DATA_DIR:-$SCRIPT_DIR/data}"
NPU_DEVICES="${NPU_DEVICES:-${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3}}"
MAX_STEPS="${MAX_STEPS:-100}"
MAX_STALENESS="${MAX_STALENESS:-1}"
if [[ ! "$MAX_STALENESS" =~ ^[1-9][0-9]*$ ]]; then
    echo 'MAX_STALENESS must be a positive integer.' >&2
    exit 1
fi
SEED="${SEED:-42}"
BENCHMARK_PROFILE="${BENCHMARK_PROFILE:-optimized}"
case "$BENCHMARK_PROFILE" in
    optimized)
        DYNAMIC_BATCHING="${DYNAMIC_BATCHING:-true}"
        DP_BALANCE="${DP_BALANCE:-true}"
        PADDING_FREE="${PADDING_FREE:-true}"
        VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-false}"
        ROLLOUT_MEMORY_FRACTION="${ROLLOUT_MEMORY_FRACTION:-0.8}"
        ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-128}"
        ;;
    baseline)
        DYNAMIC_BATCHING="${DYNAMIC_BATCHING:-false}"
        DP_BALANCE="${DP_BALANCE:-false}"
        PADDING_FREE="${PADDING_FREE:-false}"
        VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-true}"
        ROLLOUT_MEMORY_FRACTION="${ROLLOUT_MEMORY_FRACTION:-0.5}"
        ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-32}"
        ;;
    *) echo 'BENCHMARK_PROFILE must be optimized or baseline.' >&2; exit 1 ;;
esac
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-true}"
TRAIN_MAX_TOKENS_PER_GPU="${TRAIN_MAX_TOKENS_PER_GPU:-8192}"
ROLLOUT_MAX_NUM_BATCHED_TOKENS="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-8192}"
for setting in DYNAMIC_BATCHING DP_BALANCE PADDING_FREE VLLM_ENFORCE_EAGER GRADIENT_CHECKPOINTING; do
    case "${!setting}" in
        true|false) ;;
        *) echo "$setting must be true or false." >&2; exit 1 ;;
    esac
done
if [[ "$PADDING_FREE" == true ]]; then
    ATTN_IMPL="${ATTN_IMPL:-flash_attention_2}"
else
    ATTN_IMPL="${ATTN_IMPL:-sdpa}"
fi
if [[ "$DYNAMIC_BATCHING" == true ]]; then
    # Twinkle's dynamic planner also enforces a row cap. Let the token budget set
    # the actual micro-batches, while keeping 64 responses per rank per update.
    TRAIN_MICRO_BATCH_SIZE="${TRAIN_MICRO_BATCH_SIZE:-64}"
else
    TRAIN_MICRO_BATCH_SIZE="${TRAIN_MICRO_BATCH_SIZE:-2}"
fi
for setting in TRAIN_MICRO_BATCH_SIZE TRAIN_MAX_TOKENS_PER_GPU ROLLOUT_MAX_NUM_SEQS ROLLOUT_MAX_NUM_BATCHED_TOKENS; do
    if [[ ! "${!setting}" =~ ^[1-9][0-9]*$ ]]; then
        echo "$setting must be a positive integer." >&2
        exit 1
    fi
done
if (( TRAIN_MICRO_BATCH_SIZE > 64 || 64 % TRAIN_MICRO_BATCH_SIZE != 0 )); then
    echo 'TRAIN_MICRO_BATCH_SIZE must divide 64 to preserve the response GBS of 128.' >&2
    exit 1
fi
if [[ "$DYNAMIC_BATCHING" == true ]] && (( TRAIN_MAX_TOKENS_PER_GPU < 2560 )); then
    echo 'TRAIN_MAX_TOKENS_PER_GPU must cover the maximum prompt + response length (2560).' >&2
    exit 1
fi
GRADIENT_ACCUMULATION_STEPS="$((64 / TRAIN_MICRO_BATCH_SIZE))"
# Capture a small set of decode sizes to limit Ascend graph/notify resources.
# Prefill is not captured; graph sizes never exceed the configured sequence limit.
VLLM_COMPILATION_CONFIG='{}'
VERL_GRAPH_ARGS=()
if [[ "$VLLM_ENFORCE_EAGER" == false ]]; then
    graph_sizes=1
    for size in 2 4 8 16 32 64 128; do
        if (( size <= ROLLOUT_MAX_NUM_SEQS )); then
            graph_sizes+=",$size"
        fi
    done
    VLLM_COMPILATION_CONFIG="{\"cudagraph_mode\":\"FULL_DECODE_ONLY\",\"cudagraph_capture_sizes\":[$graph_sizes]}"
    VERL_GRAPH_ARGS=(
        '+actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_mode=FULL_DECODE_ONLY'
        "+actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_capture_sizes=[$graph_sizes]"
    )
fi
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$SWIFT_DIR/output/gsm8k_npu_compare}"
RUN_DIR="$OUTPUT_ROOT/${BENCHMARK_FRAMEWORK}_${BENCHMARK_PROFILE}_${RUN_TAG}"

export ASCEND_RT_VISIBLE_DEVICES="$NPU_DEVICES"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONPATH="$SWIFT_DIR:$TWINKLE_DIR/src:$VERL_DIR${PYTHONPATH:+:$PYTHONPATH}"
export TENSORBOARD_DIR="$RUN_DIR/tensorboard"

run_benchmark() {
    if [[ "${DRY_RUN:-0}" == 1 ]]; then
        printf '%q ' "$@"
        printf '\n'
        return
    fi
    "$PYTHON_BIN" - "$NPU_DEVICES" "$MODEL_PATH" "$DATA_DIR" "$MAX_STEPS" <<'PY'
import json
import sys
from pathlib import Path

devices, model, data, steps = sys.argv[1:]
ids = [int(item) for item in devices.split(',')]
if len(ids) != 4 or len(set(ids)) != 4 or ids != sorted(ids) or min(ids) < 0:
    raise ValueError('NPU_DEVICES must contain four distinct, ascending, nonnegative device IDs.')
if not Path(model).is_dir():
    raise FileNotFoundError(f'Local model directory does not exist: {model}')
manifest = json.loads((Path(data) / 'manifest.json').read_text())
if int(steps) <= 0 or int(steps) * 32 > manifest['train']['kept']:
    raise ValueError('MAX_STEPS must be positive and fit in one dataset pass: MAX_STEPS * 32 <= train rows.')
for name in ('swift_train.jsonl', 'verl_train.parquet', 'verl_test.parquet'):
    if not (Path(data) / name).is_file():
        raise FileNotFoundError(f'Run prepare_data.py first: missing {name}')
print(f'NPU devices: {devices}; training: 2; rollout: 2 (TP2); model: {model}; '
      f'steps: {steps}; response GBS: 128', flush=True)
PY
    mkdir -p "$RUN_DIR"
    {
        printf 'framework=%s; profile=%s; training_npus=2; rollout_npus=2; max_staleness=%s\n' \
            "$BENCHMARK_FRAMEWORK" "$BENCHMARK_PROFILE" "$MAX_STALENESS"
        for repo in "$SWIFT_DIR" "$TWINKLE_DIR" "$VERL_DIR"; do
            if [[ -d "$repo/.git" || -f "$repo/.git" ]]; then
                git -C "$repo" log -1 --format='%H %s'
            fi
        done
        "$PYTHON_BIN" - <<'PY'
from importlib.metadata import PackageNotFoundError, version

for package in ('torch', 'torch-npu', 'transformers', 'vllm', 'vllm-ascend', 'math-verify'):
    try:
        print(f'{package}=={version(package)}')
    except PackageNotFoundError:
        print(f'{package}: not installed')
PY
        printf '%q ' "$@"
        printf '\n'
    } > "$RUN_DIR/command.txt"
    "$@" 2>&1 | tee "$RUN_DIR/console.log"
}
