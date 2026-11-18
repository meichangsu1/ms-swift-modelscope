# NPU RL 功能测试用例

按当前本地 `dev-swift-v5-bugfix2` 的接口编写：ms-swift `bd80b4928`、Twinkle `0d5672d7`。
范围为文本 GRPO/PPO，4–8 张 NPU，训练侧使用 HF 后端和原生 FSDP2，覆盖数据并行、EP、SP，
以 GRPO 为主。这里测试功能是否执行正确；训练效率、收敛效果的对比使用
[GSM8K PK 脚本](../../../../../../examples/rl/grpo/benchmark_gsm8k_npu/README.md)。

所有硬件用例均为**待执行**。本地的 CLI/配置检查通过不等于 NPU 端到端通过。
每次执行的 `command.txt` 会记录实际代码 SHA、依赖版本、可见卡号和完整命令。

## 用例矩阵

模型列是完整远程模型名，用于明确模型身份；脚本实际加载本地 `MODEL_ROOT/模型目录名`，不会下载。
`full` 表示全量张量传输；LoRA 训练时先合并 adapter 再同步模型权重。
同步模式下实际陈旧度为 0，不额外传 `--max_staleness 0`。

| ID | 算法 | 模型 | 训练类型 | 训练并行（parallel_spec） | Sampler | 采样 TP | 部署（sampler_mode） | 同步模式（async_mode） | max_staleness | weight_sync_backend | dynamic_batching | packing_algorithm | dp_balance | parameter_sync_step | 卡数 | 是否 pass |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R01 | GRPO | `Qwen/Qwen3.5-4B` | LoRA | `fsdp4`，数据 DP=4 | vLLM-Ascend | 4 | `colocate` | `none` | 0（强制） | `full` | false | — | true | 1 | 4 | 待执行 |
| R02 | GRPO | `Qwen/Qwen3.5-4B` | Full | `fsdp2`，数据 DP=2 | vLLM-Ascend | 2 | `disaggregated` | `none` | 0（强制） | `full` | false | — | true | 1 | 4 | 待执行 |
| R03 | GRPO | `Qwen/Qwen3.5-4B` | Full | `fsdp2`，数据 DP=2 | vLLM-Ascend | 2 | `disaggregated` | `fully_async` | 1 | `delta_sharded` | false | — | true | 1 | 4 | 待执行 |
| R04 | GRPO | `Qwen/Qwen3.5-4B` | Full | `fsdp4`，数据 DP=4 | SGLang | 2 | `disaggregated` | `fully_async` | 4 | `full` | false | — | true | 1 | 6 | 待执行 |
| R05 | GRPO | `Qwen/Qwen3-4B` | Full | `fsdp2`，数据 DP=2 | SGLang | 2 | `disaggregated` | `none` | 0（强制） | `full` | false | — | true | 1 | 4 | 待执行 |
| R06 | GRPO | `Qwen/Qwen3.5-35B-A3B` | LoRA | `fsdp4ep2epfsdp2`，数据 DP=4 | vLLM-Ascend | 4 | `disaggregated` | `fully_async` | 1 | `full` | false | — | true | 1 | 8 | 待执行 |
| R07 | GRPO | `Qwen/Qwen3-30B-A3B` | Full | `fsdp8ep8epfsdp1`，数据 DP=8 | vLLM-Ascend | 8 | `colocate` | `none` | 0（强制） | `full` | false | — | true | 1 | 8 | 待执行，先确认内存容量 |
| R08 | GRPO | `Qwen/Qwen3.5-4B` | LoRA | `fsdp4ul2`＋`sequence_parallel_size=2`，数据 DP=2 | vLLM-Ascend | 2 | `disaggregated` | `fully_async` | 1 | `full` | false | — | true | 1 | 6 | 待执行 |
| R09 | PPO | `Qwen/Qwen3.5-4B` | LoRA | `fsdp4`，数据 DP=4 | vLLM-Ascend | 2 | `disaggregated` | `fully_async` | 4 | `full` | false | — | true | 1 | 6 | 待执行 |
| R10 | GRPO | `Qwen/Qwen3.5-4B` | Full | `fsdp2`，数据 DP=2 | vLLM-Ascend | 2 | `disaggregated` | `none` | 0（强制） | `full` | true | `ffd` | false | 1 | 4 | 待执行 |
| R11 | GRPO | `Qwen/Qwen3.5-4B` | Full | `fsdp2`，数据 DP=2 | vLLM-Ascend | 2 | `disaggregated` | `none` | 0（强制） | `full` | true | `kk` | true | 1 | 4 | 待执行 |
| R12 | GRPO | `Qwen/Qwen3-4B` | LoRA | `fsdp2`，数据 DP=2 | vLLM-Ascend | 2 | `disaggregated` | `fully_async` | 4 | `full` | false | — | true | 2 | 4 | 待执行 |

相比旧矩阵：R03 保留 Full＋delta；R06 改测当前的合并 LoRA 后全量同步，不再传
`weight_sync_strategy=adapter_snapshot`；R08 从“FSDP＋SP 阻塞”改为正向测试。
`one_step_off` 与 `fully_async/max_staleness=1` 使用同一流式实现，本表覆盖后者，
不为旧别名单独增加硬件用例。R10/R11 增加 DBS 两种 planner 及 DP balance 开关覆盖；
R12 覆盖稀疏发布。用例组合用于整体功能覆盖，不用于归因单个变量的性能收益。

## 公共配置与数据

全部脚本使用以下设置；各用例的差异已在表中列出：

| 参数 | 设置 |
| --- | --- |
| 数据 | 本地 GSM8K JSONL，字段 `messages`、`solution`；顺序固定，无额外切分 |
| 更新窗口 | `per_device_train_batch_size=1`、`gradient_accumulation_steps=4`；response GBS=`1 × 数据DP × 4` |
| 采样 | `num_generations=4`，`generation_batch_size=GBS/4` 个 prompt；`temperature=1`、`top_p=1`、`top_k=-1` |
| 回放 | GRPO `num_iterations=1`；PPO `num_ppo_epochs=1` |
| 长度 | `max_completion_length=512`、`max_length=1024`、`enable_thinking=false` |
| Reward | `orm=accuracy`，读取同一条数据的 `solution` |
| 精度与输入布局 | `torch_dtype=bfloat16`、`attn_impl=sdpa`、`padding_free=false`、`padding_side=right` |
| 初始化 | `model_kwargs={"memory_efficient_init":true}`，覆盖 rank 0 加载、其他 rank 配置构建路径 |
| LoRA | `lora_rank=8`、`lora_alpha=32`、`target_modules=all-linear` |
| GRPO | `loss_type=grpo`、`beta=0`；异步用例增加 `rollout_importance_sampling_mode=token_truncate` |
| PPO | `kl_coef=0.05`、`gamma=1`、`lam=0.95`；自动构建全参 critic，与 policy 共用训练卡；reference 复用关闭 adapter 的 policy |
| 优化器 | `optim=adamw_torch`、`learning_rate=1e-6`、`lr_scheduler=constant`、`weight_decay=0` |
| 运行长度 | `max_steps=8`、`num_train_epochs=1`；用于功能 smoke，不作收敛结论 |
| Sampler 内存与图 | 内存比例 `0.45`；vLLM `vllm_enforce_eager=true`，SGLang `sglang_disable_cuda_graph=true` |
| Colocate 交接 | `sleep_level=1`、`offload_model=true`、`offload_optimizer=true` |
| DBS 用例 | `max_tokens_per_micro_batch=1536`；GBS 仍按名义 B×DP×GA 决定，实际微批由 Twinkle 规划 |
| 日志与保存 | 每步记录、TensorBoard；GRPO 额外保存 completion 日志；正常结束保存 `checkpoint-final/` |

EP/UL 都不是这里额外相乘的物理卡数：R06 的训练世界是 4 张卡，R08 也是 4 张训练卡。
R08 的两个 SP peer 消费同一份数据，所以数据 DP=4/2=2，不能把 GBS 算成 16。
R07 的全参 MoE 还需容纳梯度、Adam 状态及临时张量；8 卡配置可通过校验不代表一定适配
8×61 GiB。执行前记录单卡容量和容器内存上限，OOM 单独记为资源失败，不记功能通过。

本目录不重新下载或转换数据。可复用 PK 数据准备脚本生成的 `swift_train.jsonl`：

```bash
python3 examples/rl/grpo/benchmark_gsm8k_npu/prepare_data.py \
    --model /nas/disk1/Qwen3-4B \
    --train-file /本地/gsm8k/train-00000-of-00001.parquet \
    --test-file /本地/gsm8k/test-00000-of-00001.parquet \
    --output-dir /本地/gsm8k_prepared
```

这个命令的长度过滤使用 Qwen3-4B tokenizer。不同模型功能测试共用保留后的相同题目；
不据此宣称不同模型的 token 长度或效果可直接对比。不要直接把只有 `question/answer` 的原始
Parquet 传给这些脚本，也不要使用 `remove_unused_columns=false`。

## 执行方式

每个 ID 对应同目录一个 `rXX_*.sh` 文件。先运行 R02 确认基础环境，再按采样框架、模型大小
分组执行，避免反复更换镜像和加载大模型。vLLM 与 SGLang 可以使用各自已适配的镜像，
无需为了跑这张表在同一环境安装两者；依赖版本会随运行记录。

从 ms-swift 根目录执行，例如：

```bash
NPU_DEVICES=0,1,2,3 MODEL_ROOT=/nas/disk1 \
DATASET_PATH=/本地/gsm8k_prepared/swift_train.jsonl \
bash swift/dev/tests/feature/rl/npu_functional/r02_grpo_qwen35_full_disaggregated.sh
```

R06 需要 8 张卡，也可用它复测两位数设备号的分配：

```bash
NPU_DEVICES=6,7,8,9,10,11,12,13 MODEL_ROOT=/nas/disk1 \
DATASET_PATH=/本地/gsm8k_prepared/swift_train.jsonl \
bash swift/dev/tests/feature/rl/npu_functional/r06_grpo_qwen35_moe_lora_ep.sh
```

卡号需实际空闲、容器可见、升序且不重复，数量与表中“卡数”一致。
该参数只决定进程可用的卡；Twinkle 的修复还会对 Ray 分配给**每个 worker** 的卡号重新排序。
大模型用例前可先在 Twinkle 仓库执行 CPU 回归：

```bash
python3 -m pytest -q tests/infra/test_resource_manager.py
```

脚本支持 `MODEL_PATH` 覆盖某一个模型的实际目录，`TWINKLE_DIR` 指向配套源码，
`PYTHON_BIN` 指定镜像中的 Python，以及 `MAX_STEPS`、`RUN_TAG`、`OUTPUT_ROOT`、
`MAX_COMPLETION_LENGTH`、`MAX_LENGTH`、`ROLLOUT_MEMORY_FRACTION`。
`OPTIM` 可选择优化器，默认 `adamw_torch`；`OPTIM_ARGS` 可传入优化器的额外参数。
同一 shell 中不要对所有用例统一设置 `MODEL_PATH`，否则会覆盖矩阵里的不同模型。
脚本末尾可追加尚未出现的 CLI 参数；当前 CLI 会拒绝同一参数的不同值重复出现，
已有配置请通过上述环境变量或修改对应 case 文件调整。实际完整命令以 `command.txt` 为准。

如果 R07 在创建 Adam 状态时显存不足，可临时使用无动量 SGD 验证 Full＋EP8 功能链路：

```bash
NPU_DEVICES=2,3,4,5,6,7,8,9 OPTIM=sgd OPTIM_ARGS=momentum=0 \
bash swift/dev/tests/feature/rl/npu_functional/r07_grpo_qwen3_moe_full_ep_colocate.sh
```

此时记录的优化器为 SGD，不能作为原 Adam 配置通过的证据，也不用于 Adam 的效果或性能对比。
不要在脚本末尾追加 `--optim sgd`：脚本已传入 `--optim`，CLI 会拒绝不同值的重复参数。

仅检查要运行的命令（不检查 NPU、模型、数据，也不会启动训练）：

```bash
DRY_RUN=1 bash swift/dev/tests/feature/rl/npu_functional/r08_grpo_qwen35_lora_fsdp_sp.sh
```

## 通过标准与证据

每项至少完整运行 8 个 optimizer step，退出码为 0，loss/梯度指标无 NaN/Inf，
并完成正常结束保存。reward 不要求在 8 步内单调上涨；如果所有组的 reward 完全相同，
应记录“流程通过，尚未验证非零策略梯度”，不能据此宣称有效学习。

| 用例 | 除公共标准外，需检查的行为 |
| --- | --- |
| R01 | Colocate 能连续完成采样、sleep/offload、训练、同步、下一轮采样；同步路径的 `max_partial_span` 为 0 |
| R02 | 训练与采样使用互不重叠的设备集合，至少多轮同步成功；同步路径的版本偏差为 0 |
| R03 | Full＋FSDP2 的 delta 后端真正完成首轮和后续多轮发布，采样可继续；`max_partial_span <= 1`。仅 CLI 接受参数不算通过 |
| R04 | SGLang 成功初始化并完成多轮暂停/权重更新/继续生成；`max_partial_span <= 4` |
| R05 | Qwen3＋SGLang 同步闭环成功；无重复 config 注册、旧 `enable_ep_moe` 参数透传错误 |
| R06 | 模型加载和首次 optimizer step、EP 专家通信、LoRA 合并同步、最终 adapter 保存均完成；无专家数量/参数布局错误、无无序可见卡错误 |
| R07 | EP8 全参 MoE 的加载、训练、colocate 交接和最终完整权重保存均完成；记录主机及设备内存峰值 |
| R08 | 实际 mesh 含 `ulysses_size=2`、数据 DP=2；训练前向/反向及 reference-free GRPO 路径成功，没有退回 SP=1；保存完成 |
| R09 | policy loss 与 `value_loss` 均有记录且有限，policy/critic 均完成更新；检查 `value_grad_norm`，异步偏差不超过 4 |
| R10 | token 预算驱动 FFD 微批规划；DP balance 关闭；长度不同的样本仍对应正确 advantages/logps |
| R11 | KK 微批规划和 DP balance 同时工作；检查 `dp_balance/workload_imbalance_before`、`dp_balance/workload_imbalance_after` 等指标；按输入索引恢复结果对应关系；不能要求两种 planner 的浮点结果逐位相等 |
| R12 | `parameter_sync_step=2` 时完成多轮发布，发布版本持续推进，偏差不超过 4；不能仍按每步发布来判断版本次数 |

`policy_version` 推进及 `max_partial_span` 上界是版本行为的检查，不能单独证明训练与采样
确实在时间上重叠；要确认重叠，还需保留时间戳或 profiler trace。异步偏差可以一直为 0，
不要求必须观察到正偏差。R10/R11 的采样长度是随机的；若未实际形成可变微批/不均衡输入，
应增加数据和步数，不能只凭配置为 true 就宣称该维度已覆盖。

输出目录是 `output/npu_rl_functional/RXX_时间戳/`：

- `command.txt`：版本、设备、GBS、完整参数。
- `console.log`、`tensorboard/`：运行证据；配置处理后创建的 `vN-时间戳/` 下还有框架日志。
- `vN-时间戳/checkpoint-final/`：Full 为完整 HF 权重；LoRA 为 adapter，评估时需配合原始基座模型。

请将退出码、实际完成步数、最终 checkpoint 路径、异常堆栈/trace 路径一起回填到用例结果，
不要把历史旧脚本的 pass 沿用到本表。

## 当前不列为正向用例的组合

| 配置 | 当前实现的边界 |
| --- | --- |
| `weight_sync_strategy=adapter_snapshot`、`allow_partial_rollout` | 当前 CLI 已移除，不能通过忽略未知参数继续测试 |
| `fsdp=full_shard` | 当前 preset 使用 `fsdp2` |
| `parallel_spec=fsdp4sp2` | `sp` 是 Megatron 的布尔开关；HF Ulysses 使用 `ul2` 和 `sequence_parallel_size=2` |
| PPO＋SP 或 PPO＋DBS | 当前校验拒绝，不能复用 GRPO 的 SP/DBS 参数 |
| NPU＋SGLang＋`delta_sharded` | 当前 Twinkle 的 SGLang delta collect 路径只支持 GPU；本表的 SGLang 使用 full |
| LoRA＋`delta_sharded` | delta backend 不支持 adapter；R03 必须保持 Full |
| Colocate＋异步重叠、Colocate＋delta | 当前实现要求这些路径使用分离部署 |

边界依据：`swift/dev/config/validate.py`、`swift/dev/builders/streaming.py`、
Twinkle `checkpoint_engine/manager.py`、`utils/device_mesh.py`、
`checkpoint_engine/delta/trainer/backends/fsdp.py`。配置校验仅验证这些入口约束，
NPU 内核、采样引擎版本、通信和内存适配仍需按上面的硬件用例执行。
