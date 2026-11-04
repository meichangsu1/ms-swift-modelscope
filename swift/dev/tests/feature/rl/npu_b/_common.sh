#!/usr/bin/env bash
# Shared local paths, offline preparation and logging for the seven B-owned NPU cases.

init_case() {
    CASE_ID="$1"
    EXPECTED_TRANSFORMERS="$2"
    local model_name="$3"
    local default_devices="$4"
    EXPECTED_DEVICE_COUNT="$5"
    SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
    REPO_ROOT="$(cd -- "$SCRIPT_DIR/../../../../../.." && pwd)"
    PYTHON_BIN="${PYTHON_BIN:-python}"
    MODEL_ROOT="${MODEL_ROOT:-/nas/disk1}"
    MODEL_PATH="${MODEL_PATH:-$MODEL_ROOT/$model_name}"
    DATA_ROOT="${DATA_ROOT:-/nas/disk6/ljl/data/npu_grpo}"
    PREPARED_DATA_ROOT="${PREPARED_DATA_ROOT:-$DATA_ROOT/prepared/grpo_npu}"
    OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/output/npu_grpo_b}"
    OUTPUT_DIR="${OUTPUT_DIR:-$OUTPUT_ROOT/case_$CASE_ID}"
    DRY_RUN="${DRY_RUN:-0}"
    PREPARE_ONLY="${PREPARE_ONLY:-0}"
    export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-$default_devices}"
    export USE_SWIFT_V5=1
    export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    export WANDB_MODE=offline
    export PYTHONPATH="$REPO_ROOT${TWINKLE_SRC:+:$TWINKLE_SRC}${PYTHONPATH:+:$PYTHONPATH}"
    cd -- "$REPO_ROOT"
}

prepare_dataset() {
    local kind="$1"
    local source="$2"
    local count="$3"
    PREPARED_DATASET="$PREPARED_DATA_ROOT/${kind}_${count}.jsonl"
    if [[ "$DRY_RUN" == 1 ]]; then
        printf 'Prepare %s (%s rows) from %s -> %s\n' "$kind" "$count" "$source" "$PREPARED_DATASET"
        return
    fi
    "$PYTHON_BIN" "$SCRIPT_DIR/prepare_data.py" \
        --dataset "$kind" --source "$source" --limit "$count" \
        --output-dir "$PREPARED_DATA_ROOT"
}

run_case() {
    local command=("$PYTHON_BIN" -m swift.cli.main rl "$@")
    if [[ "$DRY_RUN" == 1 ]]; then
        printf 'Case %s: Transformers=%s NPU=%s\n' "$CASE_ID" "$EXPECTED_TRANSFORMERS" "$ASCEND_RT_VISIBLE_DEVICES"
        printf '%q ' "${command[@]}"
        printf '\n'
        return
    fi
    if [[ "$PREPARE_ONLY" == 1 ]]; then
        printf 'Case %s data prepared; training was not started.\n' "$CASE_ID"
        return
    fi
    if [[ ! -d "$MODEL_PATH" || ! -f "$MODEL_PATH/config.json" ]]; then
        printf 'Local model config not found: %s/config.json; set MODEL_PATH or MODEL_ROOT.\n' "$MODEL_PATH" >&2
        return 1
    fi
    mkdir -p -- "$OUTPUT_DIR"
    "$PYTHON_BIN" - "$EXPECTED_TRANSFORMERS" "$EXPECTED_DEVICE_COUNT" "$OUTPUT_DIR" "$CASE_ID" "$MODEL_PATH" <<'PY'
import importlib.metadata
import json
import os
import platform
import sys
from pathlib import Path

expected, expected_devices, output, case, model = sys.argv[1:]
installed = importlib.metadata.version('transformers')
if installed.split('+', 1)[0] != expected:
    raise SystemExit(f'Case {case} requires transformers=={expected}; found {installed}. Activate the matching environment.')
devices = [device.strip() for device in os.environ['ASCEND_RT_VISIBLE_DEVICES'].split(',') if device.strip()]
if len(devices) != int(expected_devices) or len(devices) != len(set(devices)):
    raise SystemExit(f'Case {case} requires {expected_devices} distinct visible NPUs; found {devices}.')
packages = {}
for name in ('transformers', 'torch', 'torch-npu', 'twinkle-kit', 'vllm', 'vllm-ascend', 'sglang', 'deepspeed'):
    try:
        packages[name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        packages[name] = None
metadata = {'case': case, 'python': platform.python_version(), 'model': model,
            'visible_npus': devices, 'packages': packages}
Path(output, 'environment.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
print(json.dumps(metadata, indent=2), flush=True)
PY
    # Keep routing, device selection and shell quoting when replaying this file.
    {
        printf '#!/usr/bin/env bash\nset -euo pipefail\n'
        printf 'cd -- %q\n' "$REPO_ROOT"
        local env_name
        for env_name in USE_SWIFT_V5 ASCEND_RT_VISIBLE_DEVICES HF_HUB_OFFLINE HF_DATASETS_OFFLINE \
            TRANSFORMERS_OFFLINE WANDB_MODE PYTHONPATH; do
            printf 'export %s=%q\n' "$env_name" "${!env_name}"
        done
        printf '%q ' "${command[@]}"
        printf '\n'
    } > "$OUTPUT_DIR/command.sh"
    "${command[@]}" 2>&1 | tee "$OUTPUT_DIR/console.log"
}
