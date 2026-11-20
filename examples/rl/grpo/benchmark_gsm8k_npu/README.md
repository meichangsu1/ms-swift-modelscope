# GSM8K GRPO：四张 NPU 对比 Swift 与 verl 的异步 RL

目标是比较训练与采样重叠执行时的效率，以及相同训练预算下的效果。两份脚本均使用本地
Qwen3-4B、Full、FSDP2、分离部署：两张 NPU 训练，另两张 NPU 运行一个 TP2 采样副本，
每次实验共用四张卡。训练和采样常驻各自设备，不执行 colocate 的休眠、唤醒或借卡。

两份启动脚本现在默认使用 `BENCHMARK_PROFILE=optimized`：开启动态微批、去 padding、
DP 负载均衡和采样 decode 图，提高采样并发与显存预算。`BENCHMARK_PROFILE=baseline`
保留之前的参数组合，方便对照。优化档是性能调优起点，需要实机测量后才能判断是否最优；
它不改变 GRPO 目标、Reward、学习率、训练步数或每次更新的 response 数，也不保证准确率提升。

| 项目 | Swift | verl |
| --- | --- | --- |
| 训练 / 采样资源 | 2 / 2 张 NPU | 相同 |
| 异步训练模式 | sampler_mode=disaggregated，async_mode=fully_async | trainer_mode=separate_async，hybrid_engine=false |
| 权重同步 | 每次更新发布一次，weight_sync_backend=full | parameter_sync_step=1，checkpoint_engine.backend=nccl |
| 默认陈旧度约束 | max_staleness=1 | max_off_policy_threshold=2，strategy=drop，含义见下文 |
| 每轮 prompt / 每题 response | 32 / 4 | 32 / 4 |
| 每次更新的 response 数 | 优化档 B=64 × DP=2 × GA=1 = 128；基础档 B=2 × DP=2 × GA=32 = 128 | ppo_mini_batch_size=32 × n=4 = 128 |
| 优化档微批 | 每卡 8192 token，最多 64 条；ffd 分批 | 每卡 8192 token，动态分批 |
| Loss | grpo，delta=3 | seq-mean-token-mean，clip_ratio_c=3 |
| Old log-prob | 更新前在训练端重算，另保留采样端 log-prob | bypass_mode=false，更新前在训练端重算 |
| 重要性采样修正 | token_truncate，阈值 2.0 | rollout_is=token，阈值 2.0，不做 batch normalize |
| 精度 | FP32 主参数、BF16 前向、FP32 梯度通信 | 相同 |
| Reward | accuracy / math-verify | shared_reward.py 使用同一 scorer |
| 优化器 | AdamW，LR=1e-6，betas=(0.9, 0.999)，weight_decay=0，constant LR | 相同 |
| 训练端 offload / 梯度检查点 | offload 关闭，gradient checkpointing 开启 | 相同 |

Swift 使用当前 `dev-swift-v5` 的参数：`dp_balance`，不再传 `weight_sync_strategy` 或
`allow_partial_rollout`。verl 单独设置 `rollout.mode=async` 只表示采样引擎使用异步接口；
此处还设置 V1 的 `trainer_mode=separate_async`，让训练与采样真正重叠，并关闭 hybrid 借卡。
verl 当前 NPU checkpoint engine 注册名为 `nccl`，底层实际使用 HCCL。双方显式开启
vLLM-Ascend 的 `additional_config.rl_config.enabled=true`，适配 RL 权重更新。

## 优化档的参数

| 特性 | optimized（默认） | baseline | Swift 参数 | verl 参数 |
| --- | --- | --- | --- | --- |
| 动态微批 | true；每卡训练与 old log-prob 都用 8192 token 预算 | false；固定 2 条 | dynamic_batching / max_tokens_per_micro_batch | actor.use_dynamic_bsz / actor.ppo_max_token_len_per_gpu；rollout.log_prob_use_dynamic_bsz / rollout.log_prob_max_token_len_per_gpu |
| 去 padding | true | false | padding_free | model.use_remove_padding |
| 训练 attention | flash_attention_2（NPU 实现） | sdpa | attn_impl | model.override_config.attn_implementation |
| DP 负载均衡 | true | Swift false；verl V1 实际仍开启 | dp_balance | trainer.balance_batch，见下文 |
| 采样图 | enforce_eager=false，FULL_DECODE_ONLY | enforce_eager=true | vllm_enforce_eager / vllm_engine_kwargs.compilation_config | rollout.enforce_eager / rollout.engine_kwargs.vllm.compilation_config |
| 采样并发序列数 | 128 | 32 | vllm_max_num_seqs | rollout.max_num_seqs |
| 采样显存比例 | 0.8 | 0.5 | vllm_gpu_memory_utilization | rollout.gpu_memory_utilization |
| Prefix cache / chunked prefill | true / true | true / true | vllm_enable_prefix_caching / vllm_engine_kwargs.enable_chunked_prefill | rollout.enable_prefix_caching / rollout.enable_chunked_prefill |
| 采样每次调度 token 上限 | 8192 | 8192 | vllm_engine_kwargs.max_num_batched_tokens | rollout.max_num_batched_tokens |

表中 verl 的 `actor`、`rollout`、`model` 参数均位于 `actor_rollout_ref` 下。
Twinkle 的动态微批同时受条数上限和 token 预算限制，所以优化档将名义 B 提高到 64，GA 降到 1；
实际前后向仍按 token 预算拆分，并不是一次把 64 条长回答全部放入 NPU。verl 动态路径按 token 预算
分批，固定微批参数在该路径不决定实际批大小。两边分批与均衡算法不同，不要求微批数完全一致。

当前本地 verl V1 的 `trainer_base.py::_run_single_update` 无条件调用 `_balance_batch`，
因此 `trainer.balance_batch=false` 只复现旧启动参数，不能关闭这条路径的均衡。
比较时需记录实际执行行为，不能把 baseline 标记为“两边均未做 DP balance”。

采样图仅捕获 decode 的 `[1,2,4,8,16,32,64,128]` 档位（过滤掉超过并发上限的值），
prefill 不做图捕获，以减少 Ascend 图捕获的资源占用。`enforce_eager=false` 允许图执行，
是否实际启用仍需看 vLLM-Ascend 初始化日志。图捕获耗时计入端到端耗时，稳态吞吐另算。
配置依据见 [vLLM-Ascend graph 文档](https://docs.vllm.ai/projects/ascend/en/latest/user_guide/feature_guide/graph_mode.html)
和 [verl 性能调优文档](https://verl.readthedocs.io/en/latest/perf/perf_tuning.html)。

两边保留框架自动启用的 NPU 算子适配；本次不额外启用训练端 `torch.compile`、Liger/Triton
融合 loss、量化或借卡。它们涉及不同的依赖、精度或资源调度，不能仅设为 true 就视为有效优化。

### 陈旧度与异步调度

共享变量 `MAX_STALENESS` 默认是 1，Swift 直接使用该值；verl 使用 `MAX_STALENESS + 1`。
两边判定依据不同，这个映射只用于设定接近的异步预算：

- Swift 按「当前已发布权重版本 − 轨迹准入版本」判断，超过 `max_staleness` 的组丢弃。
- verl 按「当前训练步 − prompt 提交步 + 1」判断，超过 `max_off_policy_threshold` 的组丢弃；
  `+1` 表示区间包含提交步，因此阈值 2 允许跨一个训练步。它记录的是提交步，不是实际 token 的权重版本。

Swift 初始可准入两批，verl 设置 `num_warmup_batches=1`，在训练循环前预填一批，循环中继续提交。
两边的排队、暂停续写和过期处理实现仍有差异；不能仅凭这两个阈值就声称实际陈旧度或训练样本完全相同。
优化档与基础档的结果分开记录，不混用不同配置的耗时。

## 准备数据（仅需一次）

输入为已经离线下载的 GSM8K 原始 Parquet，必须包含 `question`、`answer` 列。
准备过程不下载模型或数据；使用本地 tokenizer 统一添加指令、关闭 thinking、过滤超过 512 token 的
prompt，并为双方输出相同顺序的数据。训练集只在此处按 seed=42 打乱一次；训练时双方不再打乱。

```bash
python3 examples/rl/grpo/benchmark_gsm8k_npu/prepare_data.py \
    --model /nas/disk1/Qwen3-4B \
    --train-file /你的数据目录/train-00000-of-00001.parquet \
    --test-file /你的数据目录/test-00000-of-00001.parquet \
    --output-dir /你的数据目录/gsm8k_compare
```

`manifest.json` 记录输入与输出的哈希、保留行数及过滤数量。测试集也使用同一长度限制，
因此评估时应报告实际保留题数，不能直接假设为完整的 1319 题。

## 分别启动

从 ms-swift 根目录运行，先后执行；每次运行前确认四张卡空闲，且上一轮进程已退出。
两个环境需要相同的 Torch、torch-npu、Transformers、vLLM-Ascend、math-verify 版本。
Twinkle、verl 默认与 ms-swift 位于同一父目录；位置不同可设置 `TWINKLE_DIR`、`VERL_DIR`。
模型、数据和仓库目录请使用绝对路径。

```bash
NPU_DEVICES=0,1,2,3 DATA_DIR=/你的数据目录/gsm8k_compare \
    bash examples/rl/grpo/benchmark_gsm8k_npu/swift.sh

NPU_DEVICES=0,1,2,3 DATA_DIR=/你的数据目录/gsm8k_compare \
    bash examples/rl/grpo/benchmark_gsm8k_npu/verl.sh
```

可覆盖 `MODEL_PATH`、`PYTHON_BIN`、`MAX_STEPS`、`MAX_STALENESS`、`SEED`、`OUTPUT_ROOT`、`RUN_TAG`、
`BENCHMARK_PROFILE`。默认模型为 `/nas/disk1/Qwen3-4B`，训练 100 次更新。
`NPU_DEVICES` 应是四个互不重复且升序排列的可用卡号。`DRY_RUN=1` 只打印命令。
例如比较更大的异步窗口时，对两份脚本都设置 `MAX_STALENESS=4`。

性能参数均可用同名环境变量覆盖，两边按同样的取值执行：

| 环境变量 | 用途 |
| --- | --- |
| DYNAMIC_BATCHING / DP_BALANCE / PADDING_FREE | 单独打开或关闭对应特性，值为 true / false |
| ATTN_IMPL | 默认随 PADDING_FREE 选择 flash_attention_2 / sdpa；去 padding 需要变长 attention |
| TRAIN_MAX_TOKENS_PER_GPU | 每卡训练和 old log-prob 的微批 token 预算，默认 8192；动态模式至少 2560 |
| TRAIN_MICRO_BATCH_SIZE | 动态模式默认条数上限 64，固定模式默认 2；必须整除 64，Swift 自动调整 GA 保持 GBS=128 |
| GRADIENT_CHECKPOINTING | 默认 true；显存足够时可比较 false 的速度 |
| VLLM_ENFORCE_EAGER | 默认优化档 false；设为 true 关闭采样图 |
| ROLLOUT_MEMORY_FRACTION | 采样显存比例 |
| ROLLOUT_MAX_NUM_SEQS | 采样并发序列数 |
| ROLLOUT_MAX_NUM_BATCHED_TOKENS | 采样每次调度的 token 上限 |

先对两边各跑 5 步，确认保存正常、loss/梯度有限、生成文本正常，再执行完整比较。例如：

```bash
MAX_STEPS=5 NPU_DEVICES=0,1,2,3 DATA_DIR=/你的数据目录/gsm8k_compare \
    bash examples/rl/grpo/benchmark_gsm8k_npu/swift.sh
MAX_STEPS=5 NPU_DEVICES=0,1,2,3 DATA_DIR=/你的数据目录/gsm8k_compare \
    bash examples/rl/grpo/benchmark_gsm8k_npu/verl.sh
```

使用旧基础配置时，在同样的启动命令前加 `BENCHMARK_PROFILE=baseline`。
训练 OOM 时先把 `TRAIN_MAX_TOKENS_PER_GPU` 从 8192 降到 4096；采样 OOM 时调整
`ROLLOUT_MEMORY_FRACTION` / `ROLLOUT_MAX_NUM_SEQS`。出现图捕获 notify/stream 资源不足时，
可先设置 `VLLM_ENFORCE_EAGER=true` 保留其他优化。每次调整都需要记录，并为两边使用相同调优预算。

## 读取结果

控制台日志与实际命令保存在 `output/gsm8k_npu_compare/<框架>_<profile>_<时间>/`。
两边都在训练结束时保存最终权重，供后续独立评估。训练期间关闭评估和中间 checkpoint 保存，
避免影响训练耗时对比：Swift 使用 `save_strategy=steps`、`save_steps=MAX_STEPS+1`，
正常结束后仍会调用最终保存；verl 使用 `trainer.save_freq=MAX_STEPS`，在最后一步保存。
两边的 TensorBoard 日志均写入本次运行目录的 `tensorboard/`。

默认训练 100 步时，最终权重的位置和使用方式为：

| 框架 | 最终 checkpoint（相对于本次运行目录） | 评估用模型 |
| --- | --- | --- |
| Swift | `vN-时间戳/checkpoint-final/` | 已保存为 HF 格式，可直接加载此目录 |
| verl | `global_step_100/actor/` | FSDP 分片，先合并导出为 HF 格式 |

verl 导出命令如下，在 verl 仓库根目录、使用训练环境执行；将运行目录和步数替换为实际值：

```bash
VERL_RUN_DIR=/绝对路径/output/gsm8k_npu_compare/verl_本次时间戳
python3 -m verl.model_merger merge \
    --backend fsdp \
    --local_dir "$VERL_RUN_DIR/global_step_100/actor" \
    --target_dir "$VERL_RUN_DIR/hf_final"
```

效率统计跳过最初 10 步和结束保存，选相同的稳态更新区间，记录：

- 完成更新数、区间墙钟耗时，以及实际参与更新的有效 response token/s；四张 NPU 全部计入资源预算。
- 采样生成 token 数、训练消费 token 数、生成长度和过期丢弃量，避免把丢弃样本的生成吞吐当成有效训练吞吐。
- trainer 等待采样的时间、权重发布时间和陈旧度分布；通过两侧时间线确认生成与训练确有重叠。
- NPU 显存峰值。端到端总耗时另列，包含初始化、最终保存和异步任务收尾，不能与稳态吞吐混算。

不要只比较 loss 或 step/s。效果可将双方 `MAX_STEPS` 改为 200，使用统一的独立评估程序与
scorer，分别加载 Swift 的 `checkpoint-final/` 和 verl 导出的 `hf_final/`；还可比较相同墙钟
预算下达到的评估准确率。

此脚本限定一个数据集 pass，启动前检查 `MAX_STEPS * 32 <= manifest.train.kept`。
这只是训练消耗的下界；异步预取和过期丢弃还会额外消耗 prompt，需留出余量，并核对两边实际完成了
`MAX_STEPS` 次更新。相同 seed、相同数据顺序不保证两个异步调度器使用相同轨迹或逐 token 一致。
