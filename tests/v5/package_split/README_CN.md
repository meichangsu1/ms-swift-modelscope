# 分包后的 NPU 端到端验收

本组只验证两件事：分包后原有训练和采样链路仍能运行；客户端和服务端不需要安装对方的包。使用本地 Qwen3-4B 和四张空闲 NPU，不下载数据集，保留 CS 采样测试，不部署或测试 DataPlane。

| 用例 | 执行环境 | 检查内容 |
| --- | --- | --- |
| `test_cs_train_save_reload` | 客户端：kit＋client；服务端：kit＋server | Swift 客户端通过真实 HTTP 完成 LoRA 训练、保存和恢复；客户端不可见 server 包，服务端不可见 client 包 |
| `test_cs_sample` | 客户端：kit＋client；服务端：kit＋server | 客户端 SDK 返回生成 token、logprob 和拼接后的输入；OpenAI 兼容接口返回 SSE 文本与结束标记，无 DataPlane 读写 |
| `test_ray_async_grpo` | 复用服务端 NPU 环境 | 原有 Swift GRPO 配方、FSDP2 trainer＋TP2 sampler、StreamingDriver 和权重同步完成四步训练，最终 adapter 有更新 |

两个 CS 用例共用服务，训练模型和采样模型各使用一张 NPU；服务退出后 GRPO 使用四张；顺序执行，不连接现有 Ray 集群，不执行全局 `ray stop`。

## 准备两个独立 Python 环境

本脚本在同一主机上使用两个独立环境，两者可以共享模型和测试输出目录，但 Python 包不能互相可见。

| 环境 | Twinkle wheel | 其他依赖 |
| --- | --- | --- |
| `CLIENT_PYTHON` | `twinkle-kit`、`twinkle-client`；没有 `twinkle-server` | 当前 ms-swift、pytest、client/core 依赖；客户端不执行 NPU 计算 |
| `SERVER_PYTHON` | `twinkle-kit`、`twinkle-server`；没有 `twinkle-client` | 当前 ms-swift、pytest、server/core 依赖及已有兼容的 Torch、torch-npu、CANN、Ray、vLLM-Ascend |

`twinkle_agentic` 已包含在 kit 中。这里验证的是两个已配置运行依赖的环境，不要求重新安装 NPU 驱动。

先准备独立的 venv 或 conda 环境。不要让它们通过 `--system-site-packages`、editable 安装或 `PYTHONPATH` 看到同时安装了三包的环境。`PACKAGE_DIST` 指向同一次构建、每包只有一个版本的 wheel 目录。将下面两个解释器路径替换成实际路径：

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

`--no-deps` 保留已有运行依赖，不补齐缺少的第三方包；core/client/server 的声明依赖仍须安装。上述命令也不会移除以前安装的对方包。如果环境不符合表中的隔离要求，测试会在启动前失败。

## 运行

在 ms-swift 根目录执行：

```bash
ASCEND_RT_VISIBLE_DEVICES=4,5,6,7 \
MODEL_PATH=/nas/disk1/Qwen3-4B \
bash tests/v5/package_split/run_npu.sh
```

`CLIENT_PYTHON`、`SERVER_PYTHON` 使用上面导出的值。将卡号换成四张空闲卡。可选 `PACKAGE_SPLIT_OUTPUT` 指定新的绝对输出目录，`CASE_TIMEOUT` 指定每组测试（CS 两项合计、GRPO 一项）的超时秒数（默认 2400）。请避免继承另一任务的 `ASCEND_VISIBLE_DEVICES` 容器过滤设置。

启动检查会分别在两个解释器中执行：

- 检查客户端没有安装、也无法找到 `twinkle_server`，服务端没有安装、也无法找到 `twinkle_client`。
- 检查所需 Twinkle 模块的路径属于对应 wheel 的文件清单，拒绝源码覆盖。
- 记录解释器、环境前缀、包版本和路径。实际导入及调用由随后执行的端到端用例验证。

脚本清除 `PYTHONPATH` 并禁用用户级 site-packages，避免运行时绕过上述环境边界。

## Server 的启动和退出

1. `CLIENT_PYTHON -m pytest` 执行 CS 用例，先进入 `cs_server` fixture。
2. fixture 再次检查客户端没有 server 包，选取空闲端口，通过 `SERVER_PYTHON serve_cs.py` 启动服务子进程。
3. 服务子进程先检查没有 client 包，然后调用 `twinkle_server.launch_server()`，创建独立的本地 Ray 集群，通过 Swift deploy builder 配置 HTTP 网关和普通 vLLM sampler，并加入训练模型；显式设置 `enable_data_plane=False`。
4. 客户端依次等待 `/api/v1/models` 和一条最小 `/api/v1/completions` 请求成功，最多等待 900 秒，确认网关和 sampler 已就绪。训练模型也必须完成加载并成功处理后续请求，测试才能通过。
5. 三次更新后保存 adapter，检查参数有限且 LoRA B 非零；再训练一次使固定输入的 logprob 改变，加载保存的 checkpoint 后检查 logprob 恢复。
6. 采样用例调用客户端 `sample()` 检查 token、logprob、输入拼接，再请求 `/api/v1/chat/completions` 的 SSE 输出，检查生成文本和 `[DONE]`。不调用 `sample_to_data_plane` 或 DataPlane 存取接口。
7. 无论成功或失败，都关闭客户端并终止、等待服务子进程退出。随后以 `SERVER_PYTHON` 运行独立的 Ray GRPO 测试，该测试不启动 CS HTTP 服务。

无需手动启动 server。客户端测试文件不导入 `twinkle_server`；只有独立服务脚本使用它。

## 结果

默认输出目录是 `output/package_split_<时间>/`：

- `client_environment.json`、`server_environment.json`：两端解释器、wheel 路径及对方包不存在的检查结果。
- `grpo_environment.json`、`environment.log`：GRPO 环境与 NPU 依赖版本。
- `server.log`、`server_config.json`：服务启动日志与实际配置。
- `cs.log` / `cs.xml`、`grpo.log` / `grpo.xml`：CS 两项和 GRPO 一项的结果。
- `cs_sample.json`：生成 token、logprob 和流式文本。
- `cs_train.json`、`server_checkpoints/`、`grpo_history.json`、`grpo/`：保存 URI、实际权重和训练指标。

只有三项均通过、没有 skip/error/failure 才打印 `All three NPU E2E cases passed`。未设置 `PACKAGE_SPLIT_E2E=1` 时，直接运行测试文件会跳过；一键脚本会设置它。

这些断言用于确认实际训练、保存恢复、CS 采样和异步 RL 链路可用，不是模型效果评测。当前范围是 Qwen3 文本、LoRA、Transformers/FSDP 与 vLLM-Ascend；其他后端或模型的原有用例可按实际发布范围继续回归。本地无 NPU 的检查不能代替服务器上的端到端运行。
----



明白，**验收就围绕这两个目标。我已调整用例，不再把 client/server 装在同一个环境里测试。**

| 目标 | 验证方式 |
|---|---|
| 分包后原有代码仍能运行 | 执行原有 Swift GRPO 配方，以及 CS 训练、保存、加载流程 |
| client/server 不相互依赖 | 使用两个独立 Python 环境，通过真实 HTTP 完成训练 |

安装要求明确为：

- **客户端环境**：`ms-swift＋twinkle-kit＋twinkle-client`，没有 `twinkle-server`。
- **服务端环境**：`ms-swift＋twinkle-kit＋twinkle-server`，没有 `twinkle-client`。

脚本会检查对方包既未安装、也无法从源码路径找到。客户端使用 `CLIENT_PYTHON`，server 由 `SERVER_PYTHON` 自动启动。Ray GRPO 复用服务端环境。

准备好两套环境后运行：

```bash
CLIENT_PYTHON=/path/to/client-env/bin/python \
SERVER_PYTHON=/path/to/server-env/bin/python \
ASCEND_RT_VISIBLE_DEVICES=4,5,6,7 \
MODEL_PATH=/nas/disk1/Qwen3-4B \
bash tests/v5/package_split/run_npu.sh
```

安装和执行说明已更新到 [README_CN.md](/Users/linjiajia/project/ms-swift/tests/v5/package_split/README_CN.md)。

本地已用真实 wheel 验证环境隔离检查，包括拒绝装入对方包、拒绝源码覆盖。修改文件检查通过；全库 lint 仍有既有问题。**实际 NPU 端到端结果仍需服务器运行确认。**




----

**不需要两个 Pod。同一个 NPU Pod 里使用两个独立的 Python 环境即可。** Annotation 1

当前脚本也是按这个方式写的：客户端通过 HTTP 请求服务端，服务端由脚本自动启动。

| 环境 | 安装内容 | 用途 |
| --- | --- | --- |
| 服务端 | kit＋server、已有 NPU 运行依赖 | 执行训练、采样和 GRPO |
| 客户端 | kit＋client、Swift 及其依赖 | 发送请求，不执行 NPU 计算 |


最省事的准备方法是：**现有 NPU 环境作为服务端，新建一个 venv 作为客户端。**

1. 在当前能正常训练的 Pod 中，设置路径：

```bash
export SERVER_PYTHON="$(python3 -c 'import sys; print(sys.executable)')"
export SWIFT_ROOT=/nas/disk6/ljl/project/dev-swift-v5/ms-swift
export PACKAGE_DIST=/实际存放三个wheel的目录

unset PYTHONPATH
export PYTHONNOUSERSITE=1
```

`PACKAGE_DIST` 指向你已经打包好的目录，每个包保留本次构建的一个版本。

2. 准备服务端环境。下面会从现有环境移除 `twinkle-client`，保留已有 Torch、torch-npu、vLLM-Ascend 等依赖：

```bash
"$SERVER_PYTHON" -m pip uninstall -y twinkle-client

"$SERVER_PYTHON" -m pip install --no-index --no-deps --force-reinstall \
  "$PACKAGE_DIST"/twinkle_kit-*.whl \
  "$PACKAGE_DIST"/twinkle_server-*.whl

"$SERVER_PYTHON" -m pip install --no-deps -e "$SWIFT_ROOT"
"$SERVER_PYTHON" -m pip install pytest
```

3. 创建独立客户端环境。**不要加 `--system-site-packages`**，否则客户端会看到现有环境里的 server 包：

```bash
CLIENT_ENV=/nas/disk6/ljl/venvs/twinkle-client-test
"$SERVER_PYTHON" -m venv "$CLIENT_ENV"
export CLIENT_PYTHON="$CLIENT_ENV/bin/python"

"$CLIENT_PYTHON" -m pip install \
  "$PACKAGE_DIST"/twinkle_kit-*.whl \
  "$PACKAGE_DIST"/twinkle_client-*.whl \
  -e "$SWIFT_ROOT" \
  pytest
```

客户端是新环境，因此这里正常安装依赖，需要可用的 pip 源或离线依赖包。它不需要安装 `twinkle-server`、vLLM-Ascend 或 torch-npu；当前 kit/Swift 仍需要 Torch 等基础依赖。

4. 启动测试：

```bash
cd "$SWIFT_ROOT"

ASCEND_RT_VISIBLE_DEVICES=4,5,6,7 \
MODEL_PATH=/nas/disk1/Qwen3-4B \
bash tests/v5/package_split/run_npu.sh
```

换成四张空闲卡即可。脚本会先检查**客户端找不到 server 包、服务端找不到 client 包**，再自动启动服务并运行三个用例。无需手动启动第二个 Pod 或 server。