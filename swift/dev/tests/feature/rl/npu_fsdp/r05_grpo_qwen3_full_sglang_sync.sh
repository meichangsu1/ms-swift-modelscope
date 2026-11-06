#!/usr/bin/env bash
# R05: GRPO, Qwen3-4B, Full, FSDP2 + SGLang TP2; four NPUs total.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"
init_case R05 Qwen3-4B 0,1,2,3 4

run_case \
    --rlhf_type grpo --num_iterations 1 --beta 0 --log_completions true \
    --tuner full --learning_rate 1e-6 \
    --nproc_per_node 2 --parallel_spec fsdp2 \
    --rollout_sampler sglang --sglang_tp_size 2 \
    --sglang_mem_fraction_static 0.5 --sglang_context_length 2048 \
    --sampler_mode disaggregated \
    --async_mode none --allow_partial_rollout false \
    --weight_sync_strategy in_place --weight_sync_backend full \
    "$@"
