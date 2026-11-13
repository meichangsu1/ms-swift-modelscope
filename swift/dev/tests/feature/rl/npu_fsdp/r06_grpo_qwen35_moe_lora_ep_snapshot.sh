#!/usr/bin/env bash
# R06: FSDP4 + EP2 + EP-FSDP2, vLLM TP4; eight NPUs total.
# Publish full model weights with LoRA merged through the checkpoint manager.
# The historical filename is retained; this no longer tests adapter snapshot publication.
# The source-rank loading and retained pre-EP snapshot require sufficient host RAM.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"
init_case R06 Qwen3.5-35B-A3B 0,1,2,3,4,5,6,7 8

run_case \
    --rlhf_type grpo --num_iterations 1 --beta 0 --log_completions true \
    --tuner lora --learning_rate 1e-4 \
    --lora_rank 8 --lora_alpha 32 --lora_dropout 0.0 --target_modules all-linear \
    --target_parameters mlp.experts.gate_up_proj mlp.experts.down_proj \
    --nproc_per_node 4 --parallel_spec fsdp4ep2epfsdp2 \
    --model_kwargs '{"memory_efficient_init":true,"fsdp_config":{"expert_parallel":{"enabled":true,"ep_size":2}}}' \
    --rollout_sampler vllm --vllm_tensor_parallel_size 4 \
    --vllm_gpu_memory_utilization 0.5 --vllm_max_model_len 2048 \
    --sampler_mode disaggregated \
    --async_mode fully_async --max_staleness 1 \
    --rollout_importance_sampling_mode token_truncate \
    --weight_sync_backend full \
    "$@"
