#!/usr/bin/env bash
# Shared settings for the four-NPU Swift / verl comparison. Source from a launcher.
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
SEED="${SEED:-42}"
ROLLOUT_MEMORY_FRACTION="${ROLLOUT_MEMORY_FRACTION:-0.5}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$SWIFT_DIR/output/gsm8k_npu_compare}"
RUN_DIR="$OUTPUT_ROOT/${BENCHMARK_FRAMEWORK}_${RUN_TAG}"

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
print(f'NPU devices: {devices}; model: {model}; steps: {steps}; response GBS: 128', flush=True)
PY
    mkdir -p "$RUN_DIR"
    {
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
