#!/usr/bin/env bash
# Common launcher for the NPU GRPO/PPO functional matrix. Source from a case script.
set -euo pipefail

CASE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SWIFT_DIR="$(cd -- "$CASE_DIR/../../../../../.." && pwd)"
TWINKLE_DIR="${TWINKLE_DIR:-$(dirname -- "$SWIFT_DIR")/twinkle}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL_ROOT="${MODEL_ROOT:-/nas/disk1}"
MODEL_PATH="${MODEL_PATH:-$MODEL_ROOT/$MODEL_NAME}"
DATASET_PATH="${DATASET_PATH:-$SWIFT_DIR/examples/rl/grpo/benchmark_gsm8k_npu/data/swift_train.jsonl}"
MAX_STEPS="${MAX_STEPS:-8}"
SAVE_STEPS="${SAVE_STEPS:-$((MAX_STEPS + 1))}"
OPTIM="${OPTIM:-adamw_torch}"
OPTIM_ARGS="${OPTIM_ARGS:-}"
SP_SIZE="${SP_SIZE:-1}"
ALGORITHM="${ALGORITHM:-grpo}"
SYNC_BACKEND="${SYNC_BACKEND:-full}"
SYNC_INTERVAL="${SYNC_INTERVAL:-1}"
DBS="${DBS:-false}"
PACKING_ALGORITHM="${PACKING_ALGORITHM:-ffd}"
DP_BALANCE="${DP_BALANCE:-true}"
ROLLOUT_MEMORY_FRACTION="${ROLLOUT_MEMORY_FRACTION:-0.45}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-512}"
MAX_LENGTH="${MAX_LENGTH:-1024}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$SWIFT_DIR/output/npu_rl_functional}"
if [[ "$OUTPUT_ROOT" != /* ]]; then
    OUTPUT_ROOT="$SWIFT_DIR/$OUTPUT_ROOT"
fi
RUN_DIR="$OUTPUT_ROOT/${CASE_ID}_$RUN_TAG"

run_case() {
    local required_devices="$TRAIN_NPUS"
    if [[ "$DEPLOYMENT" == disaggregated ]]; then
        required_devices=$((TRAIN_NPUS + SAMPLER_TP))
    fi
    local default_devices= device
    for ((device = 0; device < required_devices; device++)); do
        default_devices+="${default_devices:+,}$device"
    done
    export ASCEND_RT_VISIBLE_DEVICES="${NPU_DEVICES:-${ASCEND_RT_VISIBLE_DEVICES:-$default_devices}}"
    export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    export USE_SWIFT_V5=1 TOKENIZERS_PARALLELISM=false VLLM_WORKER_MULTIPROC_METHOD=spawn
    export PYTHONPATH="$SWIFT_DIR:$TWINKLE_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

    # SP peers share a data slice. EP changes expert placement, not the data batch size.
    local data_parallel_size=$((TRAIN_NPUS / SP_SIZE))
    local update_batch_size=$((data_parallel_size * 4))  # B=1, configured GA=4.
    local prompt_batch_size=$((update_batch_size / 4))  # Four responses per prompt.
    local -a command=(
        "$PYTHON_BIN" -m swift.dev.cli.rl
        --rlhf_type "$ALGORITHM" --model "$MODEL_PATH" --dataset "$DATASET_PATH"
        --dataset_shuffle false --train_dataloader_shuffle false --split_dataset_ratio 0
        --tuner "$TUNER" --mode ray --backend hf --fsdp fsdp2
        --parallel_spec "$PARALLEL_SPEC" --nproc_per_node "$TRAIN_NPUS"
        --sequence_parallel_size "$SP_SIZE" --padding_free false --padding_side right
        --torch_dtype bfloat16 --attn_impl sdpa --gradient_checkpointing true
        --model_kwargs '{"memory_efficient_init":true}'
        --per_device_train_batch_size 1 --gradient_accumulation_steps 4
        --generation_batch_size "$prompt_batch_size" --num_generations 4
        --dynamic_batching "$DBS" --dp_balance "$DP_BALANCE"
        --orm accuracy --enable_thinking false
        --max_completion_length "$MAX_COMPLETION_LENGTH" --max_length "$MAX_LENGTH"
        --temperature 1 --top_p 1 --top_k -1
        --sampler_mode "$DEPLOYMENT" --rollout_sampler "$SAMPLER"
        --async_mode "$ASYNC_MODE" --weight_sync_backend "$SYNC_BACKEND"
        --parameter_sync_step "$SYNC_INTERVAL"
        --optim "$OPTIM" --learning_rate 1e-6 --lr_scheduler constant
        --weight_decay 0 --max_grad_norm 1 --num_train_epochs 1 --max_steps "$MAX_STEPS"
        --seed 42 --save_strategy steps --save_steps "$SAVE_STEPS"
        --eval_strategy no --logging_steps 1 --report_to tensorboard
        --logging_dir "$RUN_DIR/tensorboard" --output_dir "$RUN_DIR"
    )
    if [[ -n "$OPTIM_ARGS" ]]; then
        command+=(--optim_args "$OPTIM_ARGS")
    fi
    if [[ "$TUNER" == lora ]]; then
        command+=(--lora_rank 8 --lora_alpha 32 --target_modules all-linear)
    fi
    if [[ "$ALGORITHM" == grpo ]]; then
        command+=(--loss_type grpo --num_iterations 1 --beta 0 --log_completions true)
        if [[ "$ASYNC_MODE" != none ]]; then
            command+=(--rollout_importance_sampling_mode token_truncate)
        fi
    else
        # The critic is full-parameter on the policy's mesh; reference reuses the LoRA base.
        command+=(--num_ppo_epochs 1 --kl_coef 0.05 --gamma 1 --lam 0.95)
    fi
    if [[ "$ASYNC_MODE" != none ]]; then
        command+=(--max_staleness "$STALENESS")
    fi
    if [[ "$DBS" == true ]]; then
        command+=(--max_tokens_per_micro_batch 1536 --packing_algorithm "$PACKING_ALGORITHM")
    fi
    if [[ "$SAMPLER" == vllm ]]; then
        command+=(
            --vllm_tensor_parallel_size "$SAMPLER_TP" --vllm_data_parallel_size 1
            --vllm_gpu_memory_utilization "$ROLLOUT_MEMORY_FRACTION" --vllm_enforce_eager true
            --vllm_max_model_len "$MAX_LENGTH" --vllm_max_num_seqs 16
            --vllm_engine_kwargs '{"seed":42,"generation_config":"vllm"}'
        )
    else
        command+=(
            --sglang_tp_size "$SAMPLER_TP" --sglang_ep_size 1
            --sglang_mem_fraction_static "$ROLLOUT_MEMORY_FRACTION"
            --sglang_context_length "$MAX_LENGTH" --sglang_disable_cuda_graph true
        )
    fi
    if [[ "$DEPLOYMENT" == colocate ]]; then
        command+=(--sleep_level 1 --offload_model true --offload_optimizer true)
    fi
    command+=("$@")
    if [[ "${DRY_RUN:-0}" == 1 ]]; then
        printf '%q ' "${command[@]}"
        printf '\n'
        return
    fi

    # Validate in a separate process, before Ray creates actors. No remote downloads.
    "$PYTHON_BIN" - "$MODEL_PATH" "$DATASET_PATH" "$required_devices" \
        "$MAX_STEPS" "$prompt_batch_size" <<'PY'
import json
import os
from pathlib import Path
import sys

model, dataset, required, steps, prompts = sys.argv[1:]
devices = [int(value) for value in os.environ['ASCEND_RT_VISIBLE_DEVICES'].split(',')]
if len(devices) != int(required) or devices != sorted(set(devices)) or min(devices) < 0:
    raise ValueError(f'NPU_DEVICES must contain exactly {required} distinct, ascending, nonnegative IDs.')
if int(steps) < 1:
    raise ValueError('MAX_STEPS must be positive.')
if not (Path(model) / 'config.json').is_file():
    raise FileNotFoundError(f'Local model config is missing: {model}/config.json')
path = Path(dataset)
if path.suffix != '.jsonl':
    raise ValueError('DATASET_PATH must be a prepared JSONL with messages and solution fields.')
count = 0
with path.open() as source:
    for line in source:
        row = json.loads(line)
        if not row.get('messages') or row.get('solution') is None or not str(row['solution']).strip():
            raise ValueError(f'Missing messages or solution at row {count + 1}.')
        count += 1
if count < int(steps) * int(prompts):
    raise ValueError('Not enough prompts for MAX_STEPS; provide more data or reduce MAX_STEPS.')
import torch
import torch_npu

if not torch.npu.is_available() or torch.npu.device_count() != int(required):
    raise RuntimeError('The configured NPU devices are not all visible to torch_npu.')
print(f'Visible NPUs: {devices}; dataset rows: {count}', flush=True)
PY
    mkdir -p "$RUN_DIR"
    {
        for repo in "$SWIFT_DIR" "$TWINKLE_DIR"; do
            git -C "$repo" log -1 --format='%H %s'
        done
        "$PYTHON_BIN" - <<'PY'
from importlib.metadata import PackageNotFoundError, version

for name in ('torch', 'torch-npu', 'transformers', 'peft', 'ray', 'vllm', 'vllm-ascend', 'sglang', 'math-verify'):
    try:
        print(f'{name}=={version(name)}')
    except PackageNotFoundError:
        print(f'{name}: not installed')
PY
        printf 'ASCEND_RT_VISIBLE_DEVICES=%s\n' "$ASCEND_RT_VISIBLE_DEVICES"
        printf 'response_gbs=%s; data_parallel_size=%s\n' "$update_batch_size" "$data_parallel_size"
        printf '%q ' "${command[@]}"
        printf '\n'
    } > "$RUN_DIR/command.txt"
    cd "$SWIFT_DIR"
    "${command[@]}" 2>&1 | tee "$RUN_DIR/console.log"
}
