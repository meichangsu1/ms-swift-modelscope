#!/usr/bin/env bash
# B case 13: Transformers 5.16.0; Qwen3-4B.
# 2 trainer NPUs (DDP) + 2 SGLang sampler NPUs (TP=2).
# USE_SWIFT_V5=1 selects the dev CLI; this script runs offline on NPU.
# DRY_RUN=1 prints the invocation; PREPARE_ONLY=1 only prepares local data.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"
init_case "13" "5.16.0" "Qwen3-4B" "0,1,2,3" "4"

prepare_dataset gsm8k "${GSM8K_DIR:-$DATA_ROOT/AI-ModelScope/gsm8k}" 256
GSM8K_DATASET="$PREPARED_DATASET"

run_case \
    --model "$MODEL_PATH" \
    --dataset "$GSM8K_DATASET" \
    --rlhf_type grpo \
    --mode ray \
    --backend hf \
    --output_dir "$OUTPUT_DIR" \
    --tuner full \
    --learning_rate 1e-6 \
    --torch_dtype bfloat16 \
    --gradient_checkpointing true \
    --max_steps "${MAX_STEPS:-20}" \
    --logging_steps 1 \
    --report_to tensorboard \
    --seed 42 \
    --data_seed 42 \
    --num_train_epochs 1 \
    --split_dataset_ratio 0 \
    --num_generations 4 \
    --temperature 1.0 \
    --num_iterations 1 \
    --generation_batch_size 8 \
    --beta 0 \
    --orm accuracy \
    --system 'Answer the question and put the final answer within \boxed{}.' \
    --nproc_per_node 2 \
    --rollout_sampler sglang \
    --sglang_tp_size 2 \
    --sglang_context_length 4096 \
    --sglang_mem_fraction_static 0.5 \
    --sampler_mode disaggregated \
    --resource_mode manual \
    --async_mode none \
    --weight_sync_strategy in_place \
    --weight_sync_backend full \
    --parameter_sync_step 1 \
    --allow_partial_rollout false \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --max_length 2048 \
    --max_completion_length 512 \
    --padding_side right \
    --padding_free false \
    --packing false \
    --use_hf false \
    --columns '{"solution":"solution"}' \
    --log_completions true \
    --eval_strategy no
