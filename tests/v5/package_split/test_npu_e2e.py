# Copyright (c) ModelScope Contributors. All rights reserved.
"""Opt-in, real NPU acceptance tests for the Twinkle package split; see README.md."""

from __future__ import annotations
import math
import os
import socket
import subprocess
import time
import urllib.error
import urllib.request
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


@pytest.fixture(scope="module")
def cs_server():
    from .check_environment import check_environment

    check_environment("client", _output() / "client_environment.json")
    server_python = os.environ["SERVER_PYTHON"]
    assert not (_output() / "server_checkpoints").exists(), "Use a fresh PACKAGE_SPLIT_OUTPUT directory"
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    base_url = f"http://127.0.0.1:{port}"
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    with (_output() / "server.log").open("w") as log:
        process = subprocess.Popen(
            [
                server_python,
                str(Path(__file__).with_name("serve_cs.py")),
                "--port",
                str(port),
                "--namespace",
                f"package-split-{uuid.uuid4().hex}",
            ],
            cwd=_output(),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            deadline = time.monotonic() + 900
            headers = {"Authorization": "Bearer package-split", "Content-Type": "application/json"}
            # The gateway can answer /models before its sampler has loaded the model.
            probes = [
                urllib.request.Request(f"{base_url}/api/v1/models", headers=headers),
                urllib.request.Request(
                    f"{base_url}/api/v1/completions",
                    headers=headers,
                    data=json.dumps({"model": "policy", "prompt": "Hello", "max_tokens": 1}).encode(),
                ),
            ]
            for request in probes:
                while time.monotonic() < deadline:
                    if process.poll() is not None:
                        raise RuntimeError(
                            f"CS server exited with code {process.returncode}; see {_output() / 'server.log'}"
                        )
                    try:
                        with urllib.request.urlopen(request, timeout=10) as response:
                            if response.status == 200:
                                break
                    except (urllib.error.URLError, OSError):
                        pass
                    time.sleep(1)
                else:
                    raise TimeoutError(f"CS server did not become ready; see {_output() / 'server.log'}")
            yield base_url
        finally:
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=90)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)


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


def test_cs_sample(cs_server, cs_client):
    """Exercise client token sampling and gateway SSE, without DataPlane storage or references."""
    import httpx
    from twinkle.data_format import InputFeature
    from twinkle_client.sampler import vLLMSampler

    from swift.dev.builders.template import build_template_from_spec, build_template_spec

    model_config, template_config = _model_configs()
    template = build_template_from_spec(build_template_spec(template_config, model_config))
    row = template.encode({"messages": [{"role": "user", "content": "Count from 1 to 5."}]}, add_generation_prompt=True)
    feature = InputFeature(input_ids=list(row["input_ids"]))
    sampler = vLLMSampler("policy")
    responses = sampler.sample([feature], sampling_params={"max_tokens": 16, "temperature": 1.0, "logprobs": 1})
    assert len(responses) == 1 and len(responses[0].sequences) == 1
    sequence = responses[0].sequences[0]
    assert sequence.tokens and len(sequence.tokens) <= 16
    assert sequence.logprobs and len(sequence.logprobs) == len(sequence.tokens)
    for token, candidates in zip(sequence.tokens, sequence.logprobs):
        assert any(token_id == token and math.isfinite(logp) for token_id, logp in candidates)
    emitted = sequence.new_input_feature
    assert emitted["input_ids"] == list(feature["input_ids"]) + list(sequence.tokens)
    assert len(emitted["input_ids"]) == len(emitted["labels"]) == len(emitted["completion_mask"])

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
    assert events and events[-1] == "[DONE]"
    chunks = [json.loads(event) for event in events[:-1]]
    assert chunks and all("error" not in chunk for chunk in chunks)
    text = "".join(
        choice.get("delta", {}).get("content") or "" for chunk in chunks for choice in chunk.get("choices", [])
    )
    assert text.strip(), "SSE completed without generated text"
    (_output() / "cs_sample.json").write_text(
        json.dumps(
            {
                "tokens": sequence.tokens,
                "logprobs": sequence.logprobs,
                "stream_text": text,
                "stream_chunks": len(chunks),
            }
        )
    )


def test_ray_async_grpo():
    """Four cards: FSDP2 trainer + TP2 sampler, real StreamingDriver and weight publication."""
    from .check_environment import check_environment

    check_environment("ray", _output() / "grpo_environment.json")
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
