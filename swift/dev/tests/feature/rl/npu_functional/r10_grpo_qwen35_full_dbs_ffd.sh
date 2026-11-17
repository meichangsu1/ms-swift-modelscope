#!/usr/bin/env bash
set -euo pipefail
CASE_ID=R10 MODEL_NAME=Qwen3.5-4B TUNER=full
TRAIN_NPUS=2 PARALLEL_SPEC=fsdp2 DBS=true PACKING_ALGORITHM=ffd DP_BALANCE=false
SAMPLER=vllm SAMPLER_TP=2 DEPLOYMENT=disaggregated ASYNC_MODE=none
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
run_case "$@"
