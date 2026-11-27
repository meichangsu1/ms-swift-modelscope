# Copyright (c) ModelScope Contributors. All rights reserved.
"""Check installed wheel boundaries without importing either side's implementation."""

from __future__ import annotations
import argparse
import importlib.metadata
import importlib.util
import sys
from pathlib import Path

import json


def check_environment(role: str, output: Path) -> dict:
    required = [("twinkle-kit", "twinkle"), ("twinkle-kit", "twinkle_agentic")]
    if role == "client":
        required.append(("twinkle-client", "twinkle_client"))
        forbidden = ("twinkle-server", "twinkle_server")
    elif role == "server":
        required.append(("twinkle-server", "twinkle_server"))
        forbidden = ("twinkle-client", "twinkle_client")
    elif role == "ray":
        # The regression can reuse the server's NPU environment; it does not use the SDK.
        forbidden = ("twinkle-client", "twinkle_client")
    else:
        raise ValueError(f"Unknown environment role: {role}")

    dist_name, module_name = forbidden
    try:
        importlib.metadata.distribution(dist_name)
    except importlib.metadata.PackageNotFoundError:
        pass
    else:
        raise RuntimeError(f"{role} environment must not have {dist_name} installed")
    if importlib.util.find_spec(module_name) is not None:
        raise RuntimeError(f"{role} environment can still find {module_name}; remove source/editable paths")

    report = {"role": role, "python": sys.executable, "prefix": sys.prefix, "absent": dist_name, "packages": {}}
    for dist_name, module_name in required:
        dist = importlib.metadata.distribution(dist_name)
        spec = importlib.util.find_spec(module_name)
        if spec is None or spec.origin is None:
            raise RuntimeError(f"{module_name} is not available in the {role} environment")
        origin = Path(spec.origin).resolve()
        relative = f"{module_name}/__init__.py"
        if relative not in {str(item) for item in dist.files or ()} or origin != dist.locate_file(relative).resolve():
            raise RuntimeError(f"{module_name} resolves outside its installed wheel: {origin}")
        report["packages"][module_name] = {"version": dist.version, "path": str(origin)}

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("role", choices=["client", "server", "ray"])
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(check_environment(args.role, args.output), indent=2))
