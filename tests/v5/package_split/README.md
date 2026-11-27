# NPU acceptance tests for the package split

This suite checks two requirements: existing training and sampling paths still run after the split, and neither the client nor the server needs the other package installed. It uses a local Qwen3-4B checkpoint and four free NPUs, with CS sampling included, no dataset download, and no DataPlane deployment or test. See [中文说明](README_CN.md).

| Test | Environment | Acceptance |
| --- | --- | --- |
| `test_cs_train_save_reload` | Client: kit + client; server: kit + server | Swift makes real HTTP training, save and reload calls; the client cannot find the server package and vice versa |
| `test_cs_sample` | Client: kit + client; server: kit + server | Client SDK returns tokens, logprobs and extended inputs; the OpenAI-compatible endpoint emits SSE text and its termination marker, without DataPlane reads or writes |
| `test_ray_async_grpo` | Reuses the server NPU environment | Existing Swift GRPO recipe, FSDP2 trainer + TP2 sampler, StreamingDriver and weight synchronization complete four steps with an updated final adapter |

The two CS tests share one server with one NPU for training and one for sampling. After server shutdown, GRPO uses four NPUs. Each phase owns its local Ray cluster; no existing cluster or global `ray stop` is used.

## Prepare separate Python environments

The runner uses two environments on the same host. Model files and test output are shared, but the Python packages must remain isolated.

| Interpreter | Twinkle wheels | Other dependencies |
| --- | --- | --- |
| `CLIENT_PYTHON` | `twinkle-kit`, `twinkle-client`; no `twinkle-server` | Current ms-swift, pytest, client/core dependencies; no client-side NPU computation |
| `SERVER_PYTHON` | `twinkle-kit`, `twinkle-server`; no `twinkle-client` | Current ms-swift, pytest, server/core dependencies and the compatible Torch/torch-npu/CANN/Ray/vLLM-Ascend runtime |

`twinkle_agentic` is included in kit. Use separate venv/conda environments with the required runtime dependencies already available. Do not inherit an all-package environment through `--system-site-packages`, editable installs, or `PYTHONPATH`. `PACKAGE_DIST` must point to one build's wheels, with only one version per package. Replace the interpreter paths below:

```bash
export CLIENT_PYTHON=/path/to/client-env/bin/python
export SERVER_PYTHON=/path/to/server-env/bin/python
SWIFT_ROOT=/nas/disk6/ljl/project/dev-swift-v5/ms-swift

"$CLIENT_PYTHON" -m pip install --no-index --no-deps --force-reinstall \
  "$PACKAGE_DIST"/twinkle_kit-*.whl "$PACKAGE_DIST"/twinkle_client-*.whl
"$SERVER_PYTHON" -m pip install --no-index --no-deps --force-reinstall \
  "$PACKAGE_DIST"/twinkle_kit-*.whl "$PACKAGE_DIST"/twinkle_server-*.whl

"$CLIENT_PYTHON" -m pip install --no-deps -e "$SWIFT_ROOT"
"$SERVER_PYTHON" -m pip install --no-deps -e "$SWIFT_ROOT"
"$CLIENT_PYTHON" -m pip install pytest
"$SERVER_PYTHON" -m pip install pytest
```

`--no-deps` preserves installed runtime versions and does not install missing third-party dependencies. These commands also do not remove previously installed opposing packages: the preflight fails if isolation is not satisfied.

## Run

From the Swift root, with the two interpreter variables exported above:

```bash
ASCEND_RT_VISIBLE_DEVICES=4,5,6,7 \
MODEL_PATH=/nas/disk1/Qwen3-4B \
bash tests/v5/package_split/run_npu.sh
```

Select four free devices. Override `PACKAGE_SPLIT_OUTPUT` with a fresh absolute output path or `CASE_TIMEOUT` with seconds per test group (both CS tests together, or GRPO; default 2400). Avoid conflicting container device filters inherited from another task.

Before execution, each interpreter checks that the opposing distribution is absent and its module cannot be found. Required Twinkle module paths must match their installed wheel file lists. Reports record interpreter, environment prefix, versions and module paths; the subsequent E2E calls exercise real imports and behavior. The runner clears `PYTHONPATH` and disables user site-packages to preserve these boundaries.

## Server lifecycle

1. Client pytest enters the `cs_server` fixture before the CS test body.
2. After checking client isolation, the fixture chooses a free port and starts `SERVER_PYTHON serve_cs.py` as a subprocess.
3. The subprocess checks server isolation, then calls `twinkle_server.launch_server()` on its own local Ray cluster. Swift's deploy builder configures a gateway and ordinary vLLM sampler with `enable_data_plane=False`; a training model is added to the same server.
4. The client waits for `/api/v1/models` and a minimal `/api/v1/completions` request to succeed, with a total 900-second startup limit. This confirms gateway and sampler readiness; the training model must also initialize and handle subsequent requests successfully.
5. The test saves after three updates and checks finite adapter values and nonzero LoRA B. One further update must change probe logprobs; checkpoint reload must restore them.
6. The sampling test calls the client's `sample()` and checks tokens, logprobs and extended inputs, then checks generated SSE text and `[DONE]` from `/api/v1/chat/completions`. It does not call `sample_to_data_plane` or DataPlane storage APIs.
7. The client and server subprocess are cleaned up on success or failure. A separate `SERVER_PYTHON` pytest process then runs Ray GRPO, without a CS HTTP server.

No manual server startup is needed. The client test does not import `twinkle_server`; only the separate server script does.

## Results

Artifacts are saved under `output/package_split_<timestamp>/`:

- `client_environment.json`, `server_environment.json`: interpreter, wheel paths and opposing-package absence.
- `grpo_environment.json`, `environment.log`: GRPO environment and NPU runtime versions.
- `server.log`, `server_config.json`: server startup and actual configuration.
- `cs.log` / `cs.xml`, `grpo.log` / `grpo.xml`: test results.
- `cs_sample.json`: generated tokens, logprobs and streamed text.
- `cs_train.json`, `server_checkpoints/`, `grpo_history.json`, `grpo/`: checkpoint URI, real weights and metrics.

All three cases must pass without skips or errors to print `All three NPU E2E cases passed`. Direct pytest runs skip unless `PACKAGE_SPLIT_E2E=1`; the runner sets it automatically.

These checks establish working training, checkpoint, CS sampling and asynchronous RL paths, not model quality. Coverage is Qwen3 text, LoRA, Transformers/FSDP and vLLM-Ascend. Run existing backend/model regressions as required by the release scope. Local checks without NPUs do not establish E2E success.
