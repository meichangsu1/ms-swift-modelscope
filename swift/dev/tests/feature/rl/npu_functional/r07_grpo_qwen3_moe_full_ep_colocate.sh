#!/usr/bin/env bash
set -euo pipefail
CASE_ID=R07 MODEL_NAME=Qwen3-30B-A3B TUNER=full
TRAIN_NPUS=8 PARALLEL_SPEC=fsdp8ep8epfsdp1
SAMPLER=vllm SAMPLER_TP=8 DEPLOYMENT=colocate ASYNC_MODE=none
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
run_case "$@"
