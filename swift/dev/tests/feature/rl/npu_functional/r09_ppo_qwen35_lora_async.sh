#!/usr/bin/env bash
set -euo pipefail
CASE_ID=R09 MODEL_NAME=Qwen3.5-4B TUNER=lora ALGORITHM=ppo
TRAIN_NPUS=4 PARALLEL_SPEC=fsdp4
SAMPLER=vllm SAMPLER_TP=2 DEPLOYMENT=disaggregated ASYNC_MODE=fully_async STALENESS=4
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
run_case "$@"
