#!/usr/bin/env bash
set -euo pipefail
CASE_ID=R01 MODEL_NAME=Qwen3.5-4B TUNER=lora
TRAIN_NPUS=4 PARALLEL_SPEC=fsdp4
SAMPLER=vllm SAMPLER_TP=4 DEPLOYMENT=colocate ASYNC_MODE=none
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
run_case "$@"
