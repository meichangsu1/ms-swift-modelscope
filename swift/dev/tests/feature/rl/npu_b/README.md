# B 负责的 GRPO NPU 测试脚本

每条用例有独立 Shell 脚本，训练参数直接写在对应文件中。脚本使用 `swift/dev`，不联网下载模型或数据，不安装或切换依赖。

| 用例 | Transformers | 模型 | 训练 / Sampler | NPU 数 | 脚本 |
| --- | --- | --- | --- | --- | --- |
| 3 | 5.8.0 | Qwen3.5-4B | Full、FSDP；vLLM，同步 | 4（2＋2） | [case_03_qwen35_full_fsdp_sync.sh](case_03_qwen35_full_fsdp_sync.sh) |
| 5 | 5.9.0 | Qwen3.5-4B | LoRA、DP×SP；vLLM，one_step_off；多图＋短视频 | 6（4＋2） | [case_05_qwen35_lora_sp_multimodal_one_step_off.sh](case_05_qwen35_lora_sp_multimodal_one_step_off.sh) |
| 9 | 5.12.0 | Qwen3.5-4B | Full、DP×SP；vLLM，同步；长序列 | 6（4＋2） | [case_09_qwen35_full_sp_sync.sh](case_09_qwen35_full_sp_sync.sh) |
| 12 | 5.16.0 | Qwen3.5-35B-A3B | LoRA、FSDP；vLLM，fully_async；单图＋多图 | 8（4＋4） | [case_12_qwen35_moe_lora_fsdp_fully_async.sh](case_12_qwen35_moe_lora_fsdp_fully_async.sh) |
| 13 | 5.16.0 | Qwen3-4B | Full、DDP；SGLang，同步 | 4（2＋2） | [case_13_qwen3_full_sglang_sync.sh](case_13_qwen3_full_sglang_sync.sh) |
| 14 | 5.16.0 | Qwen3-4B | Full、FSDP；SGLang，fully_async | 6（4＋2） | [case_14_qwen3_full_fsdp_sglang_fully_async.sh](case_14_qwen3_full_fsdp_sglang_fully_async.sh) |
| 17 | 5.22.1 | Qwen3-VL-4B-Instruct | Full、ZeRO-3；vLLM，colocate；图文混合 | 4（共享） | [case_17_qwen3_vl_full_zero3_colocate.sh](case_17_qwen3_vl_full_zero3_colocate.sh) |

建议按表中顺序执行。先激活对应 Transformers 版本的环境；脚本检查版本，不一致时会停止。vLLM 用例需准备对应的 vLLM-Ascend 环境，SGLang 用例需准备支持 NPU 的 SGLang 环境。上述组合均需实际运行确认支持情况。

从仓库根目录启动，例如：

```bash
export MODEL_ROOT=/nas/disk1
export DATA_ROOT=/nas/disk6/ljl/data/npu_grpo

bash swift/dev/tests/feature/rl/npu_b/case_03_qwen35_full_fsdp_sync.sh
```

模型默认位于 `$MODEL_ROOT/<表中模型名>`。数据目录与前面给出的 ModelScope 下载命令一致：

| 数据 | 默认本地目录 | 脚本处理 |
| --- | --- | --- |
| GSM8K | `$DATA_ROOT/AI-ModelScope/gsm8k` | 读取 `main/train*.parquet`，提取 `####` 后的最终答案 |
| NuminaMath-TIR | `$DATA_ROOT/AI-MO/NuminaMath-TIR` | 读取 `data/train*.parquet`，保留 problem、solution |
| 单图 | `$DATA_ROOT/lmms-lab/multimodal-open-r1-8k-verified` | 导出图片，保留 problem、solution |
| 多图 | `$DATA_ROOT/MUIRBENCH/MUIRBENCH` | 读取已校验的原版 Parquet，筛选双图记录，拼接问题和选项 |
| 短视频 | `$DATA_ROOT/MTEB/Video-MME_short` | 从 Parquet 导出视频，拼接 question、candidates，忽略 audio |

第 5 条采用最新约定的 ModelScope 短视频子集 `MTEB/Video-MME_short`，不需要完整 Video-MME 的视频 ZIP。选择题的参考答案转换为 `\boxed{选项字母}`，与 `--orm accuracy` 的解析方式配合。

每次启动自动准备所需条数的 JSONL，输出到 `$DATA_ROOT/prepared/grpo_npu`；已准备且源文件未变化的数据会复用。转换依赖 `pyarrow` 和 `Pillow`。输入仅包含用户问题，参考答案保留在 `solution` 中，不拼进 prompt。导出的 JSONL 使用本机绝对媒体路径；更换数据挂载路径后应重新生成。

可通过环境变量调整路径和运行方式：

| 变量 | 用途 |
| --- | --- |
| `MODEL_PATH` | 覆盖当前用例的完整本地模型路径 |
| `ASCEND_RT_VISIBLE_DEVICES` | 覆盖当前用例的 NPU 编号；数量必须与用例一致 |
| `OUTPUT_ROOT` / `OUTPUT_DIR` | 覆盖全部用例的输出根目录 / 当前用例的输出目录 |
| `PREPARED_DATA_ROOT` | 覆盖转换后数据及导出媒体的目录 |
| `GSM8K_DIR` / `NUMINA_DIR` / `SINGLE_IMAGE_DIR` / `MULTI_IMAGE_DIR` / `SHORT_VIDEO_DIR` | 分别覆盖原始 Parquet 目录，也可指向一个 Parquet 文件 |
| `TWINKLE_SRC` | 使用源码 Twinkle 时，指定其 `src` 的绝对路径；否则使用当前环境安装的 Twinkle |
| `PYTHON_BIN` | Python 可执行文件，默认 `python` |
| `MAX_STEPS` | 默认 20；临时 smoke test 可调小 |
| `DRY_RUN=1` | 只打印准备计划和完整命令，不要求已安装训练依赖或存在模型、数据 |
| `PREPARE_ONLY=1` | 只转换数据，不启动训练，不检查模型或 Transformers 版本 |

例如，仅检查第 5 条的实际命令：

```bash
DRY_RUN=1 bash swift/dev/tests/feature/rl/npu_b/case_05_qwen35_lora_sp_multimodal_one_step_off.sh
```

每条用例默认写入 `output/npu_grpo_b/case_XX`，保存 `console.log`、`environment.json` 和可复用的 `command.sh`。训练同时启用 TensorBoard 和 `log_completions`；最终训练产物按 recipe 的现有保存流程输出。

这些脚本负责启动和留存运行记录，不会自动判定全部专项通过。实际执行时需核对训练步数、有限的 loss、并行配置及异步发布/陈旧度指标。第 5/12 条的媒体续写一致性仍需要逐次 attempt 的日志或断言；现有 `partial_ratio` 和最终 completion 不能单独证明发生过中断续写或媒体保持一致。未触发或未取证的专项记为未覆盖。
