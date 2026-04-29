"""PSC environment smoke test for David & Goliath.

This checks the installed OpenRLHF/Ray/vLLM/DeepSpeed stack without launching a
large model. It also reports whether the current project code can import against
the installed OpenRLHF internal API.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import os
import platform
import sys
from importlib import metadata
from pathlib import Path


def _version(dist_name: str) -> str:
    try:
        return metadata.version(dist_name)
    except metadata.PackageNotFoundError:
        return "not-installed"


def _import(module_name: str):
    try:
        module = importlib.import_module(module_name)
        print(f"[OK] import {module_name}")
        return module
    except Exception as exc:  # pragma: no cover - diagnostic script
        print(f"[FAIL] import {module_name}: {exc.__class__.__name__}: {exc}")
        return None


def _check_versions() -> None:
    print("Python:", sys.version.replace("\n", " "))
    print("Platform:", platform.platform())
    for dist in (
        "openrlhf",
        "ray",
        "vllm",
        "deepspeed",
        "torch",
        "transformers",
        "accelerate",
    ):
        print(f"{dist}: {_version(dist)}")


def _check_cuda(require_cuda: bool) -> int:
    torch = _import("torch")
    if torch is None:
        return 1

    available = torch.cuda.is_available()
    count = torch.cuda.device_count() if available else 0
    print(f"CUDA available: {available}")
    print(f"CUDA device count: {count}")
    if available:
        print(f"Torch CUDA: {torch.version.cuda}")
        for idx in range(count):
            print(f"GPU {idx}: {torch.cuda.get_device_name(idx)}")
    elif require_cuda:
        print("[FAIL] CUDA is required for this smoke test.")
        return 1
    return 0


def _check_ray() -> int:
    ray = _import("ray")
    if ray is None:
        return 1

    started_here = False
    if not ray.is_initialized():
        try:
            print("Trying ray.init(address='auto')...")
            ray.init(address="auto", ignore_reinit_error=True)
        except Exception as exc:
            print(f"ray.init(address='auto') failed: {exc}")
            print("Starting an in-process local Ray runtime for smoke test...")
            ray.init(ignore_reinit_error=True, include_dashboard=False, num_cpus=2)
            started_here = True

    @ray.remote
    def ping() -> str:
        return "ray-ok"

    result = ray.get(ping.remote())
    print(f"Ray remote result: {result}")
    print(f"Ray resources: {ray.cluster_resources()}")
    ray.shutdown()
    if started_here:
        print("Local Ray runtime shut down.")
    return 0


def _check_openrlhf_api() -> int:
    openrlhf = _import("openrlhf")
    if openrlhf is None:
        return 1

    try:
        from openrlhf.utils import DeepspeedStrategy
    except Exception:
        try:
            from openrlhf.utils.deepspeed import DeepspeedStrategy
        except Exception as exc:
            print(f"[FAIL] DeepspeedStrategy import failed: {exc}")
            return 1

    print("DeepspeedStrategy signature:")
    print(inspect.signature(DeepspeedStrategy.__init__))

    latest_ok = True
    try:
        from openrlhf.trainer.ray.launcher import RayActorGroup
        from openrlhf.trainer.ray.ppo_actor import PolicyModelActor
        from openrlhf.trainer.ray.vllm_engine import RolloutRayActor, create_vllm_engines

        print("[OK] OpenRLHF 0.10-style Ray API imports")
        print(f"  RayActorGroup={RayActorGroup}")
        print(f"  PolicyModelActor={PolicyModelActor}")
        print(f"  RolloutRayActor={RolloutRayActor}")
        print(f"  create_vllm_engines={create_vllm_engines}")
    except Exception as exc:
        latest_ok = False
        print(f"[WARN] OpenRLHF 0.10-style Ray API import failed: {exc}")

    legacy_ok = True
    try:
        from openrlhf.trainer.ray import ActorModelRayActor, PPORayActorGroup
        from openrlhf.trainer.ray.vllm_engine import LLMRayActor

        print("[OK] legacy OpenRLHF Ray API imports")
        print(f"  ActorModelRayActor={ActorModelRayActor}")
        print(f"  PPORayActorGroup={PPORayActorGroup}")
        print(f"  LLMRayActor={LLMRayActor}")
    except Exception as exc:
        legacy_ok = False
        print(f"[WARN] legacy OpenRLHF Ray API import failed: {exc}")

    if not latest_ok and not legacy_ok:
        return 1
    return 0


def _check_project_import(strict_project_api: bool) -> int:
    here = Path(__file__).resolve()
    package_root = here.parent.parent
    project_root = package_root.parent
    for path in (project_root, package_root):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    try:
        importlib.import_module("david_and_goliath.red_team.grpo_trainer")
        print("[OK] project GRPO trainer import")
        return 0
    except Exception as exc:
        print(
            "[WARN] project GRPO trainer import failed against this OpenRLHF API: "
            f"{exc.__class__.__name__}: {exc}"
        )
        print(
            "       This means the environment installed, but the project trainer "
            "may need an OpenRLHF 0.10 API migration before full GRPO training."
        )
        return 1 if strict_project_api else 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--strict-project-api", action="store_true")
    args = parser.parse_args()

    print("PSC stack smoke test")
    print("=" * 60)
    print("RAY_ADDRESS:", os.environ.get("RAY_ADDRESS", ""))
    _check_versions()

    status = 0
    status |= _check_cuda(args.require_cuda)
    status |= _check_ray()
    status |= _check_openrlhf_api()
    status |= _check_project_import(args.strict_project_api)

    if status == 0:
        print("Smoke test completed.")
    else:
        print("Smoke test failed.")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
