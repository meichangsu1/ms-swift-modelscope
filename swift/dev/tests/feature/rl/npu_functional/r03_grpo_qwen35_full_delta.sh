#!/usr/bin/env bash
set -euo pipefail
CASE_ID=R03 MODEL_NAME=Qwen3.5-4B TUNER=full
TRAIN_NPUS=2 PARALLEL_SPEC=fsdp2 SYNC_BACKEND=delta_sharded
SAMPLER=vllm SAMPLER_TP=2 DEPLOYMENT=disaggregated ASYNC_MODE=fully_async STALENESS=1
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
run_case "$@"
