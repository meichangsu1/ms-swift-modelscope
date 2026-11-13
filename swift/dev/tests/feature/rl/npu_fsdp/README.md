# FSDP 训练侧 RL NPU 测试脚本

按最新 R01–R09 表组织，每条用例有一个 Shell 脚本。R03 按确认后的方案改为 **Full**，用于覆盖 `delta_sharded`；R09 为 **PPO＋LoRA＋异步**。所有模型仅输入文本，数据统一使用本地 `AI-ModelScope/gsm8k`。

| ID | 算法 | 模型（`Qwen/`） | 训练 | 训练并行 | Sampler / TP | 部署 | async_mode / max_staleness | 发布 / 传输 | NPU 数 | 脚本 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R01 | GRPO | Qwen3.5-4B | LoRA | FSDP4 | vLLM-Ascend / 4 | colocate | none / — | in_place / full | 4（共享） | [r01](r01_grpo_qwen35_lora_colocate.sh) |
| R02 | GRPO | Qwen3.5-4B | Full | FSDP2 | vLLM-Ascend / 2 | disaggregated | none / — | in_place / full | 4（2＋2） | [r02](r02_grpo_qwen35_full_sync.sh) |
| R03 | GRPO | Qwen3.5-4B | **Full** | FSDP2 | vLLM-Ascend / 2 | disaggregated | fully_async / 1 | in_place / delta_sharded | 4（2＋2） | [r03](r03_grpo_qwen35_full_delta_async.sh) |
| R04 | GRPO | Qwen3.5-4B | Full | FSDP4 | SGLang / 2 | disaggregated | fully_async / 4 | in_place / full | 6（4＋2） | [r04](r04_grpo_qwen35_full_sglang_async.sh) |
| R05 | GRPO | Qwen3-4B | Full | FSDP2 | SGLang / 2 | disaggregated | none / — | in_place / full | 4（2＋2） | [r05](r05_grpo_qwen3_full_sglang_sync.sh) |
| R06 | GRPO | Qwen3.5-35B-A3B | LoRA | FSDP4＋EP2＋EP-FSDP2 | vLLM-Ascend / 4 | disaggregated | fully_async / 1 | 合并 LoRA 后在线同步 / full | 8（4＋4） | [r06](r06_grpo_qwen35_moe_lora_ep_snapshot.sh) |
| R07 | GRPO | Qwen3-30B-A3B | Full | FSDP8＋EP8＋EP-FSDP1 | vLLM-Ascend / 8 | colocate | none / — | in_place / full | 8（共享） | [r07](r07_grpo_qwen3_moe_full_ep_colocate.sh) |
| R08 | GRPO | Qwen3.5-4B | LoRA | FSDP4＋SP2 | vLLM-Ascend / 2 | disaggregated | fully_async / 1 | in_place / full | 6（4＋2） | [r08](r08_grpo_qwen35_lora_fsdp_sp_async.sh) |
| R09 | PPO | Qwen3.5-4B | LoRA | FSDP4 | vLLM-Ascend / 2 | disaggregated | fully_async / 4 | in_place / full | 6（4＋2） | [r09](r09_ppo_qwen35_lora_async.sh) |

普通 FSDP 的有效 DP 为分片卡数；EP 使用相同训练卡，不额外增加卡数。脚本只设置采样 TP，未将训练 EP 自动映射为采样 EP。R09 的 critic 全参训练、与策略共用训练卡；参考前向关闭策略 adapter，不另占参考模型卡；`orm=accuracy` 不加载奖励模型。

R06 已适配上游流式 RL 重构（`78a1a1c55`）：`allow_partial_rollout` 和 `weight_sync_strategy` 参数已移除。当前用例通过 `CheckpointEngineManager.sync_weights()` 合并 LoRA 后在线同步完整模型权重，使用 `weight_sync_backend=full`；发布期间由 sampler 暂停、恢复生成。原脚本文件名中的 `_snapshot` 保留以兼容已有启动命令，但当前用例不再覆盖 `adapter_snapshot`，也不再以 `async_adapters/vN` 路径作为发布验收项。

## 使用

从仓库根目录运行，先激活匹配当前 NPU Sampler 的环境。脚本不安装依赖、不下载模型和数据，也不切换 Transformers 版本。

```bash
export MODEL_ROOT=/nas/disk1
export GSM8K_DIR=/path/to/gsm8k
# 使用源码 Twinkle 时指定；已安装对应版本则不需要。
export TWINKLE_SRC=/path/to/twinkle/src

# 先打印命令；不要求存在模型、数据或训练依赖。
DRY_RUN=1 bash swift/dev/tests/feature/rl/npu_fsdp/r02_grpo_qwen35_full_sync.sh

# 实际运行；R02 默认使用 0,1,2,3 四张卡。
bash swift/dev/tests/feature/rl/npu_fsdp/r02_grpo_qwen35_full_sync.sh

# 覆盖卡号、模型路径和步数。
ASCEND_RT_VISIBLE_DEVICES=4,5,6,7 MODEL_PATH=/models/Qwen3.5-4B MAX_STEPS=4 \
  bash swift/dev/tests/feature/rl/npu_fsdp/r03_grpo_qwen35_full_delta_async.sh
```

模型默认位于 `$MODEL_ROOT/<模型名>`。`GSM8K_DIR` 可指向含 `train*.parquet` 的目录、含 `main/train*.parquet` 的快照根目录，或单个训练 Parquet。首次启动复用 [prepare_data.py](../npu_b/prepare_data.py) 抽取 256 条，生成仅含用户问题的 `messages` 和参考答案 `solution`；`solution` 取原 `answer` 中 `####` 后的最终答案。转换只需 `pyarrow`。也可以直接用 `DATASET_PATH` 指向已准备的同格式 JSONL。

```bash
# 只准备数据，不启动模型，也不要求存在训练环境。
PREPARE_ONLY=1 bash swift/dev/tests/feature/rl/npu_fsdp/r01_grpo_qwen35_lora_colocate.sh

# 所有用例复用同一个已准备的 JSONL。
DATASET_PATH=/data/gsm8k_prompt_solution.jsonl \
  bash swift/dev/tests/feature/rl/npu_fsdp/r09_ppo_qwen35_lora_async.sh
```

| 环境变量 | 默认值 / 用途 |
| --- | --- |
| `MODEL_ROOT` / `MODEL_PATH` | `/nas/disk1` / 当前用例模型的完整本地目录 |
| `DATA_ROOT` / `GSM8K_DIR` | `/nas/disk6/ljl/data/npu_grpo` / `$DATA_ROOT/AI-ModelScope/gsm8k` |
| `DATA_LIMIT` | 256；转换时抽取的原始问题数 |
| `PREPARED_DATA_ROOT` / `DATASET_PATH` | `$DATA_ROOT/prepared/rl_npu_fsdp` / 已准备的 JSONL |
| `ASCEND_RT_VISIBLE_DEVICES` | 各脚本的默认卡号；数量必须与用例一致 |
| `PYTHON_BIN` / `TWINKLE_SRC` | `python` / 可选的 Twinkle 源码 `src` 目录 |
| `EXPECTED_TRANSFORMERS` | 可选；设置后检查版本精确匹配，否则仅记录实际版本 |
| `MAX_STEPS` | 20；执行 smoke test 可设为 4 |
| `PER_DEVICE_TRAIN_BATCH_SIZE` / `GRADIENT_ACCUMULATION_STEPS` | 1 / 4；公共训练批大小与名义梯度累积次数 |
| `SAVE_STEPS` / `SAVE_ONLY_MODEL` | 500 / true；保存间隔 / 是否仅保存模型 |
| `OUTPUT_ROOT` / `OUTPUT_DIR` | `output/npu_rl_fsdp` / 当前用例目录 |
| `DRY_RUN=1` / `PREPARE_ONLY=1` | 打印命令 / 只准备数据 |

公共批大小通过环境变量调整：

```bash
PER_DEVICE_TRAIN_BATCH_SIZE=2 GRADIENT_ACCUMULATION_STEPS=4 \
  bash swift/dev/tests/feature/rl/npu_fsdp/r02_grpo_qwen35_full_sync.sh
```

末尾可追加脚本尚未声明的 CLI 参数，例如 `--warmup_ratio 0.05`。已有参数请通过上表环境变量或编辑对应脚本调整；当前 dev CLI 拒绝同一参数重复传入不同值，不能靠末尾重复参数覆盖。

## 统一基线与记录

`_common.sh` 设置 `B=1`、`GA=4`、`num_generations=4`、`generation_batch_size=8`、`max_length=1024`、`max_completion_length=512`、`dynamic_batching=false`、`parameter_sync_step=1`、`seed=data_seed=42`。公共参数 `--fsdp fsdp2` 选择 FSDP2 实现及其预设配置（`reshard_after_forward=true`），其中的 `2` 不是卡数；训练卡数及 EP 组合由各用例的 `--parallel_spec` 决定。普通 FSDP 完整更新窗口为 `B × DP × GA`。GRPO 使用 `num_iterations=1`；PPO 使用 `num_ppo_epochs=2`。

默认关闭评估（`eval_strategy=no`），保留 TensorBoard 和每步日志。使用当前 dev 支持的 `save_strategy=steps`、`save_steps=500`，20 步测试保留结束时的模型保存；`save_only_model=true` 不保存 optimizer 状态。统一设置 `save_total_limit=2`，满足 PPO 保留上一个完整策略 / critic 检查点的要求。R06 的权重发布与训练检查点保存分开，不生成用于发布的 adapter 版本目录。如需测试完整训练状态保存，运行时设置 `SAVE_ONLY_MODEL=false SAVE_STEPS=20`。

每次执行保存到 `$OUTPUT_DIR`：

- `command.sh`：包含离线环境、可见卡号、源码路径和完整命令，可重放。
- `environment.json`：模型 / 数据路径、可见卡号及依赖版本。
- `console.log`：训练输出；训练失败会保留非零退出码。
- recipe 自身的训练日志 / TensorBoard：按框架的输出目录规则生成。

脚本不自动把启动成功记为通过。至少检查实际更新达到 `MAX_STEPS`、loss / grad_norm 有限、权重版本持续发布；异步陈旧度不超过配置上限。R03 需确认首次完整同步和后续增量发布，R06 检查合并 LoRA 后的完整权重同步及策略版本推进，R09 检查策略与 critic 都更新且 `value_loss` / `value_grad_norm` 有限。Partial rollout 未实际触发时，记录为该专项未覆盖。

## 当前限制

若 vLLM 子进程在 `check_ascend_device_type()` 报 `107001 / Invalid device ID`，可先运行不加载模型的设备探针，复用 R06 的训练 4 卡、采样 4 卡布局：

```bash
ASCEND_RT_VISIBLE_DEVICES=6,7,8,9,10,11,12,13 \
  python swift/dev/tests/feature/rl/npu_fsdp/diagnose_vllm_npu.py \
  > /tmp/r06_npu_probe.log 2>&1
```

探针使用独立的本地 Ray 集群，输出主进程的设备检查、Twinkle 分配的 sampler 卡号、sampler actor 的设备检查，以及新建子进程中导入 vLLM-Ascend 并调用设备类型检查的结果。模型组只保留资源布局，不加载模型，也不启动训练。子进程不是实际的 vLLM EngineCore / TP worker，检查通过后仍需排查真实引擎启动过程中对环境或设备状态的修改。

- **R08 阻塞**：当前 `swift/dev/config/validate.py::_check_sp_hf_feasibility()` 拒绝 FSDP＋Ulysses SP。脚本使用 `fsdp=fsdp2` 和 `sequence_parallel_size=2`，实际启动会报校验错误；不能将该错误或改成 SP=1 后跑通记作 R08 通过。
- **R06 主机内存**：脚本启用 `memory_efficient_init=true`，但当前 EP 分发流程会克隆并保留 `_rank0_pre_ep_full_state_dict`。35B 模型的完整 FP32 参数约 131 GiB，加载阶段原模型与快照并存，200 GiB 容器仍可能 OOM。需要先解决该加载 / 快照生命周期问题或提供足够主机内存，再执行此项。
- **R07 资源待确认**：保持当前 colocate 初始化行为，没有开启源 rank＋meta 加载；多个训练 rank 各自加载完整 FP32 权重，需要较大主机内存。NPU 训练峰值也需要实测；`offload_model` / `offload_optimizer` 只作用于训练与采样的阶段交接，不代表训练期间持续 CPU offload。
- **R03 传输依赖**：NPU 增量后端使用自己的 HCCL / vLLM-Ascend 适配，需目标环境具备对应依赖；它与普通 `full` 传输后端的原生 HCCL 实现不同。不能只用 `full` 同步成功代替增量链路验证。
