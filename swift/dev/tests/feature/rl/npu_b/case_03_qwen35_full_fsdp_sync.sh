#!/usr/bin/env bash
# B case 03: Transformers 5.8.0; Qwen3.5-4B.
# 2 trainer NPUs (FSDP) + 2 sampler NPUs (TP=2).
# USE_SWIFT_V5=1 selects the dev CLI; this script runs offline on NPU.
# DRY_RUN=1 prints the invocation; PREPARE_ONLY=1 only prepares local data.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"
init_case "03" "5.8.0" "Qwen3.5-4B" "0,1,2,3" "4"

prepare_dataset single_image "${SINGLE_IMAGE_DIR:-$DATA_ROOT/lmms-lab/multimodal-open-r1-8k-verified}" 256
SINGLE_IMAGE_DATASET="$PREPARED_DATASET"

run_case \
    --model "$MODEL_PATH" \
    --dataset "$SINGLE_IMAGE_DATASET" \
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
    --fsdp full_shard \
    --parallel_spec fsdp2 \
    --rollout_sampler vllm \
    --vllm_tensor_parallel_size 2 \
    --vllm_gpu_memory_utilization 0.5 \
    --vllm_max_model_len 4096 \
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
    --language_model_only false \
    --max_pixels 262144 \
    --vllm_limit_mm_per_prompt '{"image":1}' \
    --vit_gradient_checkpointing true \
    --use_hf false \
    --remove_unused_columns false \
    --columns '{"solution":"solution"}' \
    --log_completions true \
    --eval_strategy no
