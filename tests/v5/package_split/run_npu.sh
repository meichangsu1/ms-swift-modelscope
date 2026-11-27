#!/usr/bin/env bash
# Run from the NPU environment with the split packages installed from the current checkouts.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "${ROOT}"
: "${ASCEND_RT_VISIBLE_DEVICES:?Set four free NPU IDs, for example 4,5,6,7}"
export MODEL_PATH=${MODEL_PATH:-/nas/disk1/Qwen3-4B}
export PACKAGE_SPLIT_E2E=1
export PACKAGE_SPLIT_OUTPUT=${PACKAGE_SPLIT_OUTPUT:-"${PWD}/output/package_split_$(date +%Y%m%d_%H%M%S)"}
export RAY_ADDRESS=local
export RAY_DEDUP_LOGS=0
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
PYTHON=${PYTHON:-python3}
CASE_TIMEOUT=${CASE_TIMEOUT:-2400}
[[ "${PACKAGE_SPLIT_OUTPUT}" = /* ]] || { echo 'PACKAGE_SPLIT_OUTPUT must be absolute' >&2; exit 1; }
[[ ! -e "${PACKAGE_SPLIT_OUTPUT}" ]] || { echo 'Use a fresh PACKAGE_SPLIT_OUTPUT directory' >&2; exit 1; }
mkdir -p "${PACKAGE_SPLIT_OUTPUT}"
command -v timeout >/dev/null || { echo 'Linux timeout command is required' >&2; exit 1; }

# Record installed distributions and import origins. No sys.path/PYTHONPATH patch for split packages.
"${PYTHON}" - <<'PY' | tee "${PACKAGE_SPLIT_OUTPUT}/environment.log"
import importlib
import importlib.metadata
import os
from pathlib import Path

import torch
import torch_npu

assert Path(os.environ['MODEL_PATH']).is_dir(), os.environ['MODEL_PATH']
assert torch.npu.is_available() and torch.npu.device_count() == 4, 'Expose exactly four free NPUs'
for dist, module in [('ms-swift', 'swift'), ('twinkle-kit', 'twinkle'),
                     ('twinkle-client', 'twinkle_client'), ('twinkle-server', 'twinkle_server')]:
    print(dist, importlib.metadata.version(dist), importlib.import_module(module).__file__)
for name in ['torch', 'torch-npu', 'transformers', 'ray', 'vllm', 'vllm-ascend', 'pytest']:
    print(name, importlib.metadata.version(name))
for key in ['MODEL_PATH', 'ASCEND_RT_VISIBLE_DEVICES', 'ASCEND_VISIBLE_DEVICES', 'RAY_ADDRESS', 'PYTHONPATH']:
    print(f'{key}={os.environ.get(key)}')
PY

TEST=tests/v5/package_split/test_npu_e2e.py
# CS tests share their own server; the RL test starts in a fresh interpreter/Ray cluster.
timeout --signal=INT --kill-after=120 "${CASE_TIMEOUT}" "${PYTHON}" -m pytest "${TEST}" \
    -k 'test_cs_' -vv -s --tb=long --junitxml="${PACKAGE_SPLIT_OUTPUT}/cs.xml" \
    2>&1 | tee "${PACKAGE_SPLIT_OUTPUT}/cs.log"
timeout --signal=INT --kill-after=120 "${CASE_TIMEOUT}" "${PYTHON}" -m pytest "${TEST}" \
    -k 'test_ray_async_grpo' -vv -s --tb=long --junitxml="${PACKAGE_SPLIT_OUTPUT}/grpo.xml" \
    2>&1 | tee "${PACKAGE_SPLIT_OUTPUT}/grpo.log"

"${PYTHON}" - <<'PY'
import os
import xml.etree.ElementTree as ET
from pathlib import Path

output = Path(os.environ['PACKAGE_SPLIT_OUTPUT'])
for name, expected in [('cs', 2), ('grpo', 1)]:
    cases = ET.parse(output / f'{name}.xml').findall('.//testcase')
    assert len(cases) == expected, (name, len(cases))
    assert not any(case.find(tag) is not None for case in cases for tag in ['failure', 'error', 'skipped']), name
print(f'All three NPU E2E cases passed. Logs and checkpoints: {output}')
PY
