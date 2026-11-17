# GSM8K GRPO：四张 NPU 对比 Swift 与 verl

两份脚本均使用本地 Qwen3-4B、Full、FSDP2、同步训练、colocate。
四张卡同时承担训练与采样，采样为两个 TP2 副本，整个实验只需四张卡。

| 项目 | Swift | verl |
| --- | --- | --- |
| 每轮 prompt / 每题 response | 32 / 4 | 32 / 4 |
| 每次更新的 response 数 | B=2 × DP=4 × GA=16 = 128 | ppo_mini_batch_size=32 × n=4 = 128 |
| 每个 rank 的固定微批 | 2 条 response | 2 条 response |
| Loss | grpo，delta=3 | seq-mean-token-mean，clip_ratio_c=3 |
| Old log-prob | 采样端 log-prob | bypass_mode=true，使用采样端 log-prob |
| 精度 | FP32 主参数、BF16 前向、FP32 梯度通信 | 相同 |
| Reward | accuracy / math-verify | shared_reward.py 使用同一 scorer |
| 优化器 | AdamW，LR=1e-6，betas=(0.9, 0.999)，weight_decay=0，constant LR | 相同 |
| DBS / DP balance / graph / 训练端 offload | 关闭 | 关闭 |

Swift 使用当前 `dev-swift-v5` 的参数：`dp_balance`，不再传 `weight_sync_strategy` 或
`allow_partial_rollout`。verl 的 `rollout.mode=async` 是引擎接口模式，训练仍为 `trainer_mode=sync`。
verl 当前 NPU checkpoint engine 注册名为 `nccl`，底层实际使用 HCCL。

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

可覆盖 `MODEL_PATH`、`PYTHON_BIN`、`MAX_STEPS`、`SEED`、`OUTPUT_ROOT`、`RUN_TAG`、
`ROLLOUT_MEMORY_FRACTION`。默认模型为 `/nas/disk1/Qwen3-4B`，训练 100 次更新。
`NPU_DEVICES` 应是四个互不重复且升序排列的可用卡号。`DRY_RUN=1` 只打印命令。

## 读取结果

控制台日志与实际命令保存在 `output/gsm8k_npu_compare/<框架>_<时间>/`。
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

效率统计跳过最初 10 步和结束保存，比较相同稳态区间的耗时、有效 response token/s、生成长度、
显存峰值。不要只比较 loss 或 step/s。效果可将双方 `MAX_STEPS` 改为 200，使用统一的独立评估
程序与 scorer，分别加载 Swift 的 `checkpoint-final/` 和 verl 导出的 `hf_final/`。
此脚本限定一个数据集 pass，要求 `MAX_STEPS * 32 <= manifest.train.kept`，避免数据耗尽导致
两边实际更新次数不同。相同 seed 不保证两个调度器生成逐 token 一致的轨迹。
