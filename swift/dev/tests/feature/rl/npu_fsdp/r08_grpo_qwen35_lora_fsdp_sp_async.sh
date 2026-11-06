#!/usr/bin/env bash
# R08 is BLOCKED: current dev validation rejects FSDP + Ulysses SP.
# Keep the requested combination; a validation failure is not an end-to-end pass.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"
init_case R08 Qwen3.5-4B 0,1,2,3,4,5 6

run_case \
    --rlhf_type grpo --num_iterations 1 --beta 0 --log_completions true \
    --tuner lora --learning_rate 1e-4 \
    --lora_rank 8 --lora_alpha 32 --lora_dropout 0.0 --target_modules all-linear \
    --nproc_per_node 4 --parallel_spec fsdp4 --sequence_parallel_size 2 \
    --rollout_sampler vllm --vllm_tensor_parallel_size 2 \
    --vllm_gpu_memory_utilization 0.5 --vllm_max_model_len 2048 \
    --sampler_mode disaggregated \
    --async_mode fully_async --max_staleness 1 --allow_partial_rollout true \
    --rollout_importance_sampling_mode token_truncate \
    --weight_sync_strategy in_place --weight_sync_backend full \
    "$@"
