# NPU acceptance tests for the package split

Run real end-to-end paths first, followed by wheel/install and unit checks. This suite uses a local **Qwen3-4B** checkpoint, four free NPUs on one host, and the existing compatible vLLM-Ascend environment. No dataset download is needed. See [中文说明](README_CN.md) for the full commands and acceptance criteria.

| Test | Coverage | Acceptance |
| --- | --- | --- |
| `test_cs_train_save_reload` | Swift remote wrapper/tuner/processor builders → client HTTP → server training controller → training worker → Swift loader and shared TemplateSpec | Save after three updates, nonzero finite LoRA B; another update changes logprobs, then checkpoint reload restores them |
| `test_cs_sample_and_data_plane` | Swift deployment builder → independent server → vLLM-Ascend → token sampling, SSE streaming, and DataPlaneClient | Nonempty tokens, aligned feature lengths, completed SSE stream, two stored samples with logprobs, reward write/read and DataRef release |
| `test_ray_async_grpo` | Swift GRPO → FSDP2 trainer + TP2 sampler, reward worker, StreamingDriver and weight publication | Four steps, finite losses, advancing policy versions, bounded spans, updated adapter in final checkpoint |

Register all three packages from the current checkout in the existing NPU environment:

```bash
TWINKLE_ROOT=/nas/disk6/ljl/project/dev-swift-v5/twinkle
SWIFT_ROOT=/nas/disk6/ljl/project/dev-swift-v5/ms-swift
python3 -m pip install --no-deps -e "$TWINKLE_ROOT" \
  -e "$TWINKLE_ROOT/twinkle_cs/client" -e "$TWINKLE_ROOT/twinkle_cs/server" -e "$SWIFT_ROOT"
python3 -m pip install pytest
```

`--no-deps` preserves the installed NPU stack; missing client/server dependencies must already be installed according to their pyproject files. From the Swift root, select four free devices and run:

```bash
ASCEND_RT_VISIBLE_DEVICES=4,5,6,7 \
MODEL_PATH=/nas/disk1/Qwen3-4B \
bash tests/v5/package_split/run_npu.sh
```

The CS tests share a server using two cards. GRPO then runs in a new interpreter on all four cards. Each group owns its local Ray cluster; no global `ray stop` is used. The runner records distribution versions/import origins, logs, JUnit XML, checkpoints, generated samples, and training history under `output/package_split_<timestamp>/`. Skips do not count as success. Override `PYTHON`, `PACKAGE_SPLIT_OUTPUT` (absolute path), or `CASE_TIMEOUT` (seconds per group, default 2400) as needed. Do not inherit conflicting container device filters from another task.

To rerun one group, set `PACKAGE_SPLIT_E2E=1`, `MODEL_PATH`, a fresh `PACKAGE_SPLIT_OUTPUT`, `RAY_ADDRESS=local`, and the visible devices, then use `python3 -m pytest tests/v5/package_split/test_npu_e2e.py -k test_cs_ -vv -s --tb=long` or `-k test_ray_async_grpo`. Without the opt-in variable the tests skip.

The CS training test drives Swift builders and its remote wrapper, not the full SFT CLI. These cases do not cover Tinker SDK, WebUI, SGLang, Megatron, multimodal models, or every model family. GRPO uses a content-dependent smoke-test reward to produce gradients, not to measure math accuracy.

After E2E passes, build/check the three wheels and sdists, test isolated kit / kit+client / kit+server / all-package installs and `pip check`, and rerun this suite with all three wheels. Then run the Twinkle client/server/template and Swift processor/builder/deploy unit tests. Syntax/static checks on a development host without NPUs do not establish E2E success.
