# 分包后的 NPU 端到端验收

先运行真实链路，再验证 wheel、独立安装和单元测试。本组针对 Qwen3-4B、单机四张空闲 NPU、vLLM-Ascend；使用本地模型，不下载数据集。需要沿用已有兼容的 Torch、torch-npu、CANN 和 vLLM 环境。

| 用例 | 真实执行的链路 | 通过条件 |
| --- | --- | --- |
| `test_cs_train_save_reload` | Swift 远程模型封装、LoRA/processor builder → `twinkle_client` HTTP → `twinkle_server` 训练控制器 → Twinkle 训练 worker → Swift model loader、共享 `TemplateSpec`、`DevInputProcessor` | 三次反向与更新后保存，LoRA B 有非零有限值；再训练一步改变权重，重新加载 `twinkle://` checkpoint 后固定输入 logprob 恢复 |
| `test_cs_sample_and_data_plane` | Swift deploy 配置构建 → 独立服务进程 → vLLM-Ascend → 客户端 token 采样、SSE 流式输出和 DataPlaneClient | 生成非空 token，训练字段长度一致；流式输出正常结束；异步生成两条记录，读取 logprob，写回 reward 后再次读取并释放 DataRef |
| `test_ray_async_grpo` | Swift GRPO recipe → FSDP2 训练 worker、TP2 sampler、reward worker、`twinkle_agentic` StreamingDriver、checkpoint engine | 四个训练 step，loss 有限，policy version 递增，版本跨度不超过 1，最终 checkpoint 中 LoRA B 有更新 |

前两个用例共用一个自动创建的 CS 服务，各占一张 NPU；第三个用例在新的 Python 进程中使用四张卡。不会连接现有 Ray 集群，也不执行 `ray stop --force`。

## 安装当前源码

在服务器将 ms-swift 和 Twinkle 都切到本次分包代码。已有训练依赖的环境中，重新登记三个分包，避免仍导入旧安装。下面的 `--no-deps` 保留已有 NPU 依赖，**不负责补齐缺少的依赖**；例如未安装服务端依赖时，须先按 `twinkle_cs/server/pyproject.toml` 补齐。

```bash
TWINKLE_ROOT=/nas/disk6/ljl/project/dev-swift-v5/twinkle
SWIFT_ROOT=/nas/disk6/ljl/project/dev-swift-v5/ms-swift
python3 -m pip install --no-deps -e "$TWINKLE_ROOT" \
  -e "$TWINKLE_ROOT/twinkle_cs/client" -e "$TWINKLE_ROOT/twinkle_cs/server" -e "$SWIFT_ROOT"
python3 -m pip install pytest
```

## 启动

确认这四张卡空闲，然后在 ms-swift 根目录运行：

```bash
ASCEND_RT_VISIBLE_DEVICES=4,5,6,7 \
MODEL_PATH=/nas/disk1/Qwen3-4B \
bash tests/v5/package_split/run_npu.sh
```

`MODEL_PATH` 可以换成另一处 **Qwen3-4B** 目录；本组模型类型和模板固定为 `qwen3`。可选 `PYTHON` 指定解释器、`PACKAGE_SPLIT_OUTPUT` 指定绝对输出目录、`CASE_TIMEOUT` 指定每组超时秒数（默认 2400）。请避免继承另一任务的 `ASCEND_VISIBLE_DEVICES` 容器过滤设置；只暴露这四张测试卡。

默认输出为 `output/package_split_<时间>/`：

- `environment.log`：分包版本、实际导入路径、依赖版本和设备环境。
- `server.log`、`server_config.json`：CS 服务和 worker 日志、实际部署配置。
- `cs.log` / `cs.xml`、`grpo.log` / `grpo.xml`：测试结果；异常立即失败，skip 不能算通过。
- `cs_train.json`、`cs_sample.json`、`server_checkpoints/`：远程保存 URI、采样结果、服务端 adapter。
- `grpo_history.json`、`grpo/`：训练指标和最终 adapter checkpoint。

单独重跑时设置 `PACKAGE_SPLIT_E2E=1`、`MODEL_PATH`、`PACKAGE_SPLIT_OUTPUT`（使用新目录）和 `RAY_ADDRESS=local`，再执行：

```bash
python3 -m pytest tests/v5/package_split/test_npu_e2e.py -k test_cs_ -vv -s --tb=long
python3 -m pytest tests/v5/package_split/test_npu_e2e.py -k test_ray_async_grpo -vv -s --tb=long
```

未设置 `PACKAGE_SPLIT_E2E=1` 时，本文件默认跳过，避免普通测试收集意外启动服务。

## 通过后再验证

这三个用例验证选中链路中的实际导入和跨进程调用，不证明全部可选后端都可用。CS 训练使用 Swift 公共 builder 和远程封装，不是完整的 SFT CLI 配方；还未覆盖 Tinker SDK、WebUI、SGLang、Megatron、多模态和所有模型家族。GRPO 的内容相关 reward 用于触发真实梯度，不用于评价数学能力。

端到端通过后，再构建三个分包的 sdist/wheel，运行 `twine check`；在独立环境中验证 kit、kit+client、kit+server、三包组合的安装、导入与 `pip check`，并在三包 wheel 环境中复跑上述 E2E。这样可以区分源码调用链问题与漏打包、缺依赖等安装问题。最后运行 Twinkle 的 client/server/template 单元测试与 Swift 的 processor/builder/deploy 配置测试。

本地开发机无 NPU 时，只能检查脚本语法和静态代码；必须以服务器实际测试结果作为验收依据。
