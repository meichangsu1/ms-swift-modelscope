#!/usr/bin/env bash
# R02: GRPO, Qwen3.5-4B, Full, FSDP2 + vLLM TP2; four NPUs total.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"
init_case R02 Qwen3.5-4B 0,1,2,3 4

run_case \
    --rlhf_type grpo --num_iterations 1 --beta 0 --log_completions true \
    --tuner full --learning_rate 1e-6 \
    --nproc_per_node 2 --parallel_spec fsdp2 \
    --rollout_sampler vllm --vllm_tensor_parallel_size 2 \
    --vllm_gpu_memory_utilization 0.5 --vllm_max_model_len 2048 \
    --sampler_mode disaggregated \
    --async_mode none --allow_partial_rollout false \
    --weight_sync_strategy in_place --weight_sync_backend full \
    "$@"
