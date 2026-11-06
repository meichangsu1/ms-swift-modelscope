#!/usr/bin/env bash
# R09: PPO LoRA policy + full-parameter critic share FSDP4; vLLM TP2 is disjoint.
# The policy with its adapter disabled supplies the reference; six NPUs total.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"
init_case R09 Qwen3.5-4B 0,1,2,3,4,5 6

run_case \
    --rlhf_type ppo --num_ppo_epochs 2 --log_completions false \
    --gamma 1.0 --lam 0.95 --kl_coef 0.05 --cliprange 0.2 --cliprange_value 0.2 \
    --tuner lora --learning_rate 1e-5 \
    --lora_rank 8 --lora_alpha 32 --lora_dropout 0.0 --target_modules all-linear \
    --nproc_per_node 4 --parallel_spec fsdp4 \
    --rollout_sampler vllm --vllm_tensor_parallel_size 2 \
    --vllm_gpu_memory_utilization 0.5 --vllm_max_model_len 2048 \
    --sampler_mode disaggregated \
    --async_mode fully_async --max_staleness 4 --allow_partial_rollout true \
    --weight_sync_strategy in_place --weight_sync_backend full \
    "$@"
