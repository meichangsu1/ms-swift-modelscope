#!/usr/bin/env bash
# R04: GRPO, Qwen3.5-4B, Full, FSDP4 + SGLang TP2; six NPUs total.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"
init_case R04 Qwen3.5-4B 0,1,2,3,4,5 6

run_case \
    --rlhf_type grpo --num_iterations 1 --beta 0 --log_completions true \
    --tuner full --learning_rate 1e-6 \
    --nproc_per_node 4 --parallel_spec fsdp4 \
    --rollout_sampler sglang --sglang_tp_size 2 \
    --sglang_mem_fraction_static 0.5 --sglang_context_length 2048 \
    --sampler_mode disaggregated \
    --async_mode fully_async --max_staleness 4 --allow_partial_rollout true \
    --rollout_importance_sampling_mode token_truncate \
    --weight_sync_strategy in_place --weight_sync_backend full \
    "$@"
