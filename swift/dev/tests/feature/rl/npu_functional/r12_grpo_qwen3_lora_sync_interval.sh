#!/usr/bin/env bash
set -euo pipefail
CASE_ID=R12 MODEL_NAME=Qwen3-4B TUNER=lora
TRAIN_NPUS=2 PARALLEL_SPEC=fsdp2 SYNC_INTERVAL=2
SAMPLER=vllm SAMPLER_TP=2 DEPLOYMENT=disaggregated ASYNC_MODE=fully_async STALENESS=4
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
run_case "$@"
