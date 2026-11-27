# Copyright (c) ModelScope Contributors. All rights reserved.
"""Opt-in, real NPU acceptance tests for the Twinkle package split; see README.md."""

from __future__ import annotations
import asyncio
import math
import multiprocessing
import os
import socket
import uuid
from pathlib import Path

import json
import pytest

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(os.environ.get("PACKAGE_SPLIT_E2E") != "1", reason="Use package_split/run_npu.sh"),
]


def _output() -> Path:
    path = Path(os.environ["PACKAGE_SPLIT_OUTPUT"]).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _model_configs():
    from swift.dev.config import ModelConfig, TemplateConfig

    return (
        ModelConfig(model=os.environ["MODEL_PATH"], model_type="qwen3", torch_dtype="bfloat16", attn_impl="sdpa"),
        TemplateConfig(template="qwen3", max_length=512, enable_thinking=False),
    )


def _serve(port: int, namespace: str) -> None:
    # A fresh interpreter must resolve the split packages and Swift's worker hook itself.
    output = _output()
    with (output / "server.log").open("w") as log:
        os.dup2(log.fileno(), 1)
        os.dup2(log.fileno(), 2)
    os.environ["TWINKLE_DEFAULT_SAVE_DIR"] = str(output / "server_checkpoints")
    os.environ["TWINKLE_WORKER_SETUP_HOOK"] = "swift.dev.processor.server_plugin:register"
    os.chdir(output)

    import ray
    from twinkle_server import launch_server
    from twinkle_server.config import ApplicationSpec

    from swift.dev.config import DeployConfig
    from swift.dev.recipe.infer.run_deploy import build_server_config

    model_config, template_config = _model_configs()
    config = build_server_config(
        model_config,
        template_config,
        deploy_config=DeployConfig(
            host="127.0.0.1",
            port=port,
            route_prefix="/api/v1",
            api_key="package-split",
            served_model_name="policy",
            ray_namespace=namespace,
            enable_data_plane=True,
        ),
        engine_args={
            "enforce_eager": True,
            "max_model_len": 512,
            "gpu_memory_utilization": 0.4,
            "enable_lora": True,
            "max_lora_rank": 8,
            "additional_config": {"weight_nz_mode": 0},
        },
    )
    config.applications.append(
        ApplicationSpec(
            name="training-policy",
            route_prefix="/api/v1/model/policy",
            import_path="model",
            args={
                "backend": "transformers",
                "model_id": model_config.model,
                "model_loader": "swift.dev.model.loader:build_model_loader",
                "device_group": {"name": "model", "ranks": 1, "device_type": "npu"},
                "device_mesh": {"dp_size": 1, "device_type": "npu"},
                "min_replicas": 1,
                "max_replicas": 1,
                "max_loras": 1,
                "max_r": 8,
                "target_modules": ["q_proj", "v_proj"],
                "max_length": 512,
                "mixed_precision": "bf16",
                "adapter_config": {"adapter_timeout": 1800},
            },
        )
    )
    config.trust_remote_code = True  # Resolve this checkout's explicitly configured Swift loader.
    # Validate after adding the training deployment, before starting any workers.
    config = type(config).model_validate(config.model_dump())
    (output / "server_config.json").write_text(config.model_dump_json(indent=2))
    try:
        launch_server(config=config, ray_address="local")
    finally:
        ray.shutdown()


@pytest.fixture(scope="module")
def cs_server():
    from swift.dev.recipe.infer.run_deploy import _wait_until_accessible

    assert not (_output() / "server_checkpoints").exists(), "Use a fresh PACKAGE_SPLIT_OUTPUT directory"
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    process = multiprocessing.get_context("spawn").Process(
        target=_serve, args=(port, f"package-split-{uuid.uuid4().hex}")
    )
    process.start()
    try:
        _wait_until_accessible(
            port,
            900,
            process,
            "/api/v1",
            api_key="package-split",
            ready_probe=(
                f"http://127.0.0.1:{port}/api/v1/completions",
                {
                    "model": "policy",
                    "prompt": "Hello",
                    "max_tokens": 1,
                },
            ),
        )
        yield f"http://127.0.0.1:{port}"
    finally:
        process.terminate()
        process.join(timeout=90)
        if process.is_alive():
            process.kill()
            process.join(timeout=10)


@pytest.fixture
def cs_client(cs_server):
    from twinkle_client import init_twinkle_client

    client = init_twinkle_client(base_url=cs_server, api_key="package-split")
    try:
        yield client
    finally:
        client.close()


def test_cs_train_save_reload(cs_server, cs_client):
    """Swift builders -> HTTP -> server -> Swift loader/TemplateSpec/processor -> real backward."""
    from safetensors.torch import load_file

    from swift.dev.builders.remote import RemoteModel, apply_remote_tuner, configure_remote_processor
    from swift.dev.builders.template import build_template_from_spec, build_template_spec
    from swift.dev.config import DistributedConfig, TunerConfig

    local_model, template_config = _model_configs()
    distributed = DistributedConfig(mode="remote", server_base_url=cs_server, server_api_key="package-split")
    model = RemoteModel(cs_client.model("policy"))
    try:
        apply_remote_tuner(
            model,
            TunerConfig(tuner="lora", lora_rank=8, lora_alpha=16, target_modules=["q_proj", "v_proj"]),
            gradient_accumulation_steps=1,
        )
        configure_remote_processor(model, template_config, local_model, distributed)
        model.set_loss("CrossEntropyLoss")
        model.set_optimizer("AdamW", lr=1e-4)
        template = build_template_from_spec(build_template_spec(template_config, local_model))
        batch = [
            template.encode(
                {
                    "messages": [
                        {"role": "user", "content": "What is 2 + 2?"},
                        {"role": "assistant", "content": "2 + 2 = 4."},
                    ]
                }
            )
        ]
        for _ in range(3):
            model.forward_backward(inputs=batch)
            model.clip_grad_and_step()
            model.zero_grad()
        saved = model.save("split-checkpoint", save_optimizer=True)
        assert saved.twinkle_path.startswith("twinkle://"), saved
        # Read real saved adapter parameters: successful HTTP replies alone are insufficient.
        weights = list((_output() / "server_checkpoints").rglob("adapter_model.safetensors"))
        assert weights, "No adapter checkpoint was written by the server"
        import numpy as np

        tensors = [tensor for path in weights for tensor in load_file(str(path)).values()]
        assert all(tensor.isfinite().all() for tensor in tensors), "Non-finite saved adapter"
        assert any(
            "lora_B" in key and tensor.abs().max() > 0
            for path in weights
            for key, tensor in load_file(str(path)).items()
        ), "Saved LoRA B remains zero"

        def logps():
            values = np.asarray(model.forward_only(inputs=batch)["logps"], dtype=float)
            assert values.size and np.isfinite(values).all() and (values < 0).any()
            return values

        before = logps()
        # Change the live adapter before reload so the restore assertion cannot pass as a no-op.
        model.forward_backward(inputs=batch)
        model.clip_grad_and_step()
        model.zero_grad()
        assert not np.array_equal(logps(), before), "The extra optimizer step did not change the probe"
        model.load(saved.twinkle_path)
        np.testing.assert_allclose(logps(), before, rtol=0, atol=1e-4)
        (_output() / "cs_train.json").write_text(json.dumps({"checkpoint": saved.twinkle_path, "steps": 3}))
    finally:
        model.close()


def test_cs_sample_and_data_plane(cs_server, cs_client):
    """SSE/token sampling + async DataRef round trip, with real vLLM-Ascend."""
    import httpx
    from twinkle.data_format import InputFeature
    from twinkle_client import DataPlaneClient
    from twinkle_client.sampler import vLLMSampler

    from swift.dev.builders.template import build_template_from_spec, build_template_spec

    model_config, template_config = _model_configs()
    template = build_template_from_spec(build_template_spec(template_config, model_config))
    row = template.encode({"messages": [{"role": "user", "content": "Count from 1 to 5."}]}, add_generation_prompt=True)
    feature = InputFeature(input_ids=list(row["input_ids"]))
    sampler = vLLMSampler("policy")
    params = {"max_tokens": 16, "temperature": 1.0, "logprobs": 1}
    response = sampler.sample([feature], sampling_params=params)
    assert len(response) == 1 and response[0].sequences[0].tokens
    sequence = response[0].sequences[0]
    emitted = sequence.new_input_feature
    assert emitted["input_ids"] == list(feature["input_ids"]) + list(sequence.tokens)
    assert len(emitted["input_ids"]) == len(emitted["labels"]) == len(emitted["completion_mask"])

    # Exercise the streaming queue bridge moved from the server package into twinkle.sampler.
    with httpx.stream(
        "POST",
        f"{cs_server}/api/v1/chat/completions",
        timeout=300,
        headers={"Authorization": "Bearer package-split"},
        json={
            "model": "policy",
            "messages": [{"role": "user", "content": "Count to five."}],
            "max_tokens": 16,
            "stream": True,
        },
    ) as response:
        response.raise_for_status()
        events = [line[6:] for line in response.iter_lines() if line.startswith("data: ")]
    assert events and events[-1] == "[DONE]", events
    chunks = [json.loads(event) for event in events[:-1]]
    assert chunks and all("error" not in chunk for chunk in chunks), chunks
    assert any(choice.get("delta", {}).get("content") for chunk in chunks for choice in chunk.get("choices", []))

    async def round_trip():
        data = DataPlaneClient()
        ref = await sampler.asample_to_data_plane(
            [feature], sampling_params=params, num_samples=2, group_ids=["split-group"]
        )
        try:
            rows = await data.aget(ref, fields=["decoded", "tokens", "sampled_logprobs"])
            assert len(rows) == 2
            for item in rows:
                assert item["tokens"] and len(item["tokens"]) == len(item["sampled_logprobs"])
                assert all(math.isfinite(value) for value in item["sampled_logprobs"])
            ref = await data.aappend(ref, [{"reward": 1.0}, {"reward": 0.0}])
            scored = await data.aget(ref, fields=["reward"])
            assert [item["reward"] for item in scored] == [1.0, 0.0]
            (_output() / "cs_sample.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2))
        finally:
            await data.arelease(ref)

    asyncio.run(round_trip())


def test_ray_async_grpo():
    """Four cards: FSDP2 trainer + TP2 sampler, real StreamingDriver and weight publication."""
    from tests.v5.feature.rl.conftest import lora_b_nonzero, rl_configs, run_rl
    from tests.v5.feature.rl.test_async_e2e import _varied_content_reward

    dataset = _output() / "prompts.jsonl"
    prompts = ["Name three animals.", "Write a short greeting.", "Name three colours.", "Count to five."]
    dataset.write_text(
        "".join(json.dumps({"messages": [{"role": "user", "content": prompt}]}) + "\n" for prompt in prompts * 16)
    )
    configs = rl_configs(
        rl_type="grpo",
        model=os.environ["MODEL_PATH"],
        model_type="qwen3",
        template="qwen3",
        dataset=str(dataset),
        out_dir=str(_output() / "grpo"),
        nproc=2,
        max_steps=4,
        model_over={"attn_impl": "sdpa"},
        template_over={"max_length": 512, "enable_thinking": False},
        dist_over={"fsdp": "fsdp2", "parallel_spec": "fsdp2"},
        train_over={"gradient_accumulation_steps": 2},
        rlhf_over={"num_generations": 4, "beta": 0.0, "max_completion_length": 32},
        reward_over={"orm": [_varied_content_reward]},
        generation_over={"temperature": 1.0, "max_new_tokens": 32},
        rollout_over={
            "sampler_mode": "disaggregated",
            "async_mode": "fully_async",
            "max_staleness": 1,
            "vllm_tensor_parallel_size": 2,
            "rollout_importance_sampling_mode": "token_truncate",
            "vllm_engine_kwargs": {"additional_config": {"weight_nz_mode": 0}},
        },
    )
    history = run_rl(configs)
    (_output() / "grpo_history.json").write_text(json.dumps(history, indent=2))
    assert len(history) >= 4, history
    assert all(math.isfinite(step["loss"]) for step in history)
    versions = [step["policy_version"] for step in history]
    assert versions[-1] > versions[0], versions
    for step in history:
        assert 0 <= step["version_span_mean"] <= 1
        assert 0 <= step["max_partial_span"] <= 1
    checkpoint = Path(configs["checkpoint_config"].output_dir) / "checkpoint-final"
    assert (checkpoint / "args.json").is_file()
    assert lora_b_nonzero(str(checkpoint)), "The GRPO optimizer did not update the saved adapter"
