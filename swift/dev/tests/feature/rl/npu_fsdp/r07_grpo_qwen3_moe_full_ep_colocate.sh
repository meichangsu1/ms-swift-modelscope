#!/usr/bin/env bash
# R07: Full, FSDP8 + EP8 + EP-FSDP1, colocated vLLM TP8; eight NPUs.
# Host loading and training-device peaks must be measured before marking this case passed.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"
init_case R07 Qwen3-30B-A3B 0,1,2,3,4,5,6,7 8

run_case \
    --rlhf_type grpo --num_iterations 1 --beta 0 --log_completions true \
    --tuner full --learning_rate 1e-6 \
    --nproc_per_node 8 --parallel_spec fsdp8ep8epfsdp1 \
    --model_kwargs '{"fsdp_config":{"expert_parallel":{"enabled":true,"ep_size":8}}}' \
    --rollout_sampler vllm --vllm_tensor_parallel_size 8 \
    --vllm_gpu_memory_utilization 0.5 --vllm_max_model_len 2048 \
    --sampler_mode colocate \
    --sleep_level 1 --offload_model true --offload_optimizer true \
    --async_mode none --allow_partial_rollout false \
    --weight_sync_strategy in_place --weight_sync_backend full \
    "$@"
