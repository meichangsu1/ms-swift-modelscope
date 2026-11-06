#!/usr/bin/env bash
# R01: GRPO, Qwen3.5-4B, LoRA, FSDP4; vLLM TP4 on the same four NPUs.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"
init_case R01 Qwen3.5-4B 0,1,2,3 4

run_case \
    --rlhf_type grpo --num_iterations 1 --beta 0 --log_completions true \
    --tuner lora --learning_rate 1e-4 \
    --lora_rank 8 --lora_alpha 32 --lora_dropout 0.0 --target_modules all-linear \
    --nproc_per_node 4 --parallel_spec fsdp4 \
    --rollout_sampler vllm --vllm_tensor_parallel_size 4 \
    --vllm_gpu_memory_utilization 0.5 --vllm_max_model_len 2048 \
    --sampler_mode colocate \
    --sleep_level 1 --offload_model true --offload_optimizer true \
    --async_mode none --allow_partial_rollout false \
    --weight_sync_strategy in_place --weight_sync_backend full \
    "$@"
