#!/usr/bin/env bash
set -euo pipefail
CASE_ID=R06 MODEL_NAME=Qwen3.5-35B-A3B TUNER=lora
TRAIN_NPUS=4 PARALLEL_SPEC=fsdp4ep2epfsdp2
SAMPLER=vllm SAMPLER_TP=4 DEPLOYMENT=disaggregated ASYNC_MODE=fully_async STALENESS=1
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
run_case "$@"
