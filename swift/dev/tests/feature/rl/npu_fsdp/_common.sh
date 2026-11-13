#!/usr/bin/env bash
# Shared offline paths, baseline parameters and run records for R01-R09.

init_case() {
    CASE_ID="$1"
    local model_name="$2"
    local default_devices="$3"
    EXPECTED_DEVICE_COUNT="$4"
    SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
    REPO_ROOT="$(cd -- "$SCRIPT_DIR/../../../../../.." && pwd)"
    PYTHON_BIN="${PYTHON_BIN:-python}"
    MODEL_ROOT="${MODEL_ROOT:-/nas/disk1}"
    MODEL_PATH="${MODEL_PATH:-$MODEL_ROOT/$model_name}"
    DATA_ROOT="${DATA_ROOT:-/nas/disk6/ljl/data/npu_grpo}"
    GSM8K_DIR="${GSM8K_DIR:-$DATA_ROOT/AI-ModelScope/gsm8k}"
    DATA_LIMIT="${DATA_LIMIT:-256}"
    PREPARED_DATA_ROOT="${PREPARED_DATA_ROOT:-$DATA_ROOT/prepared/rl_npu_fsdp}"
    CUSTOM_DATASET="${DATASET_PATH:+1}"
    DATASET_PATH="${DATASET_PATH:-$PREPARED_DATA_ROOT/gsm8k_$DATA_LIMIT.jsonl}"
    OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/output/npu_rl_fsdp}"
    OUTPUT_DIR="${OUTPUT_DIR:-$OUTPUT_ROOT/$CASE_ID}"
    MAX_STEPS="${MAX_STEPS:-20}"
    DRY_RUN="${DRY_RUN:-0}"
    PREPARE_ONLY="${PREPARE_ONLY:-0}"
    export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-$default_devices}"
    export USE_SWIFT_V5=1
    export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    export WANDB_MODE=offline
    export PYTHONPATH="$REPO_ROOT${TWINKLE_SRC:+:$TWINKLE_SRC}${PYTHONPATH:+:$PYTHONPATH}"
    cd -- "$REPO_ROOT"
}

prepare_gsm8k() {
    # An explicitly prepared JSONL can be shared by all nine cases.
    if [[ "$CUSTOM_DATASET" == 1 ]]; then
        if [[ ! -f "$DATASET_PATH" ]]; then
            printf 'Prepared dataset not found: %s. Set DATASET_PATH to an existing JSONL.\n' "$DATASET_PATH" >&2
            return 1
        fi
        return
    fi
    "$PYTHON_BIN" "$SCRIPT_DIR/../npu_b/prepare_data.py" \
        --dataset gsm8k --source "$GSM8K_DIR" \
        --limit "$DATA_LIMIT" --output-dir "$PREPARED_DATA_ROOT"
}

record_environment() {
    "$PYTHON_BIN" - "$EXPECTED_DEVICE_COUNT" "$OUTPUT_DIR" "$CASE_ID" "$MODEL_PATH" "$DATASET_PATH" <<'PY'
import importlib.metadata
import json
import os
import platform
import sys
from pathlib import Path

expected_devices, output, case, model, dataset = sys.argv[1:]
if not Path(model, 'config.json').is_file():
    raise SystemExit(f'Local model config not found: {model}/config.json. Set MODEL_PATH or MODEL_ROOT.')
if not Path(dataset).is_file():
    raise SystemExit(f'Prepared dataset not found: {dataset}.')
devices = [item.strip() for item in os.environ['ASCEND_RT_VISIBLE_DEVICES'].split(',') if item.strip()]
if len(devices) != int(expected_devices) or len(devices) != len(set(devices)):
    raise SystemExit(f'Case {case} requires {expected_devices} distinct visible NPUs; found {devices}.')
packages = {}
for name in ('transformers', 'torch', 'torch-npu', 'ray', 'twinkle-kit', 'peft', 'vllm', 'vllm-ascend', 'sglang'):
    try:
        packages[name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        packages[name] = None
expected_transformers = os.environ.get('EXPECTED_TRANSFORMERS')
if expected_transformers and (packages['transformers'] or '').split('+', 1)[0] != expected_transformers:
    raise SystemExit(f"Expected transformers=={expected_transformers}; found {packages['transformers']}.")
metadata = {'case': case, 'python': platform.python_version(), 'model': model, 'dataset': dataset,
            'visible_npus': devices, 'packages': packages}
Path(output, 'environment.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
print(json.dumps(metadata, indent=2), flush=True)
PY
}

run_case() {
    # Each case supplies its algorithm, parallelism, sampler and publication parameters.
    # Extra arguments may add fields; dev rejects conflicting duplicate flags.
    local command=(
        "$PYTHON_BIN" -m swift.cli.main rl
        --model "$MODEL_PATH"
        --dataset "$DATASET_PATH"
        --mode ray --backend hf
        --fsdp fsdp2
        --torch_dtype bfloat16
        --attn_impl sdpa
        --gradient_checkpointing true
        --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
        --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-4}"
        --num_generations 4
        --generation_batch_size 8
        --num_train_epochs 1
        --temperature 1.0
        --max_length 1024
        --max_completion_length 512
        --padding_side right --padding_free false --packing false
        --orm accuracy
        --system 'Solve step by step and put the final answer within \boxed{}.'
        --resource_mode manual
        --parameter_sync_step 1
        --dynamic_batching false
        --max_steps "$MAX_STEPS"
        --logging_steps 1 --report_to tensorboard
        --eval_strategy no --save_strategy steps --save_steps "${SAVE_STEPS:-500}"
        --save_only_model "${SAVE_ONLY_MODEL:-true}" --save_total_limit 2
        --split_dataset_ratio 0
        --seed 42 --data_seed 42
        --use_hf false
        --columns '{"solution":"solution"}'
        --output_dir "$OUTPUT_DIR"
        "$@"
    )
    printf 'Case %s: NPU=%s\n' "$CASE_ID" "$ASCEND_RT_VISIBLE_DEVICES"
    if [[ "$DRY_RUN" == 1 ]]; then
        printf 'Dataset: %s (prepare from %s if absent)\n' "$DATASET_PATH" "$GSM8K_DIR"
        printf '%q ' "${command[@]}"
        printf '\n'
        return
    fi
    prepare_gsm8k
    if [[ "$PREPARE_ONLY" == 1 ]]; then
        printf 'Dataset ready: %s; training was not started.\n' "$DATASET_PATH"
        return
    fi
    mkdir -p -- "$OUTPUT_DIR"
    record_environment
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
    # Case scripts enable pipefail, so a training failure remains a nonzero exit.
    "${command[@]}" 2>&1 | tee "$OUTPUT_DIR/console.log"
}
