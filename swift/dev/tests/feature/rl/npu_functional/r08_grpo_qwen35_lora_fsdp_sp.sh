#!/usr/bin/env bash
set -euo pipefail
CASE_ID=R08 MODEL_NAME=Qwen3.5-4B TUNER=lora
TRAIN_NPUS=4 PARALLEL_SPEC=fsdp4ul2 SP_SIZE=2
SAMPLER=vllm SAMPLER_TP=2 DEPLOYMENT=disaggregated ASYNC_MODE=fully_async STALENESS=1
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
run_case "$@"
