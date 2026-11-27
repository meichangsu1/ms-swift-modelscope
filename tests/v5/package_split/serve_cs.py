# Copyright (c) ModelScope Contributors. All rights reserved.
"""CS training and sampling server, executed with SERVER_PYTHON; no DataPlane deployment."""

from __future__ import annotations
import argparse
import os
from pathlib import Path

from check_environment import check_environment


def serve(port: int, namespace: str) -> None:
    output = Path(os.environ["PACKAGE_SPLIT_OUTPUT"]).resolve()
    check_environment("server", output / "server_environment.json")
    os.environ["TWINKLE_DEFAULT_SAVE_DIR"] = str(output / "server_checkpoints")
    os.environ["TWINKLE_WORKER_SETUP_HOOK"] = "swift.dev.processor.server_plugin:register"
    os.chdir(output)

    import ray
    from twinkle_server import launch_server
    from twinkle_server.config import ApplicationSpec

    from swift.dev.config import DeployConfig, ModelConfig, TemplateConfig
    from swift.dev.recipe.infer.run_deploy import build_server_config

    config = build_server_config(
        ModelConfig(model=os.environ["MODEL_PATH"], model_type="qwen3", torch_dtype="bfloat16", attn_impl="sdpa"),
        TemplateConfig(template="qwen3", max_length=512, enable_thinking=False),
        deploy_config=DeployConfig(
            host="127.0.0.1",
            port=port,
            route_prefix="/api/v1",
            api_key="package-split",
            served_model_name="policy",
            ray_namespace=namespace,
            enable_data_plane=False,
            num_replicas=1,
        ),
        engine_args={
            "enforce_eager": True,
            "max_model_len": 512,
            "gpu_memory_utilization": 0.4,
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
                "model_id": os.environ["MODEL_PATH"],
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
    # Resolve this checkout's explicitly configured Swift loader in server workers.
    config.trust_remote_code = True
    config = type(config).model_validate(config.model_dump())
    (output / "server_config.json").write_text(config.model_dump_json(indent=2))
    try:
        launch_server(config=config, ray_address="local")
    finally:
        ray.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--namespace", required=True)
    args = parser.parse_args()
    serve(args.port, args.namespace)
