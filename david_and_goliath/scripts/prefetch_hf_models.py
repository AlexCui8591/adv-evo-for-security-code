#!/usr/bin/env python3
"""Prefetch Hugging Face model snapshots for Bridges-2 jobs.

The script is intentionally separate from training. Run it inside a GPU-shared
interactive or batch job, then run training with HF_HUB_OFFLINE=1 if you want
to force cache-only behavior.
"""

from __future__ import annotations

import argparse
import fcntl
import os
import re
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover - handled with a clear runtime error.
    yaml = None

try:
    from huggingface_hub import snapshot_download
except ImportError:  # pragma: no cover - handled in main().
    snapshot_download = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROFILE_MODELS = {
    "smoke": ["gpt2"],
    "red8b": ["unsloth/Llama-3.1-8B-Instruct"],
    "blue32b": ["Qwen/Qwen2.5-Coder-32B-Instruct"],
    "blue32": ["Qwen/Qwen2.5-Coder-32B-Instruct"],
}
PROFILE_MODELS["red"] = PROFILE_MODELS["red8b"]
PROFILE_MODELS["blue"] = PROFILE_MODELS["blue32b"]
PROFILE_MODELS["active"] = PROFILE_MODELS["red8b"] + PROFILE_MODELS["blue32b"]
PROFILE_MODELS["all"] = PROFILE_MODELS["active"]

ACTIVE_CONFIG_PATTERN = "david_and_goliath/configs/experiment/coevo_8b_local_blue_32b.yaml"

MODEL_KEYS = {
    "model",
    "model_name",
    "base_model",
    "repo_id",
    "served_model_name",
}

NON_HF_PREFIXES = (
    "gpt-",
    "claude",
    "sonnet",
    "us.anthropic.",
    "anthropic/",
    "openai/",
    "http://",
    "https://",
    "/",
    ".",
)

HF_REPO_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._/-]*$")

DEFAULT_IGNORE_PATTERNS = (
    "*.h5",
    "*.msgpack",
    "*.onnx",
    "*.ot",
    "*.tflite",
    "flax_model.*",
    "model.onnx*",
    "tf_model.*",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        action="append",
        choices=sorted([*PROFILE_MODELS.keys(), "config"]),
        help="Model group to prefetch. Repeatable. Default: active.",
    )
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help="Additional Hugging Face model repo id. Repeatable.",
    )
    parser.add_argument(
        "--config",
        action="append",
        default=[],
        help="YAML config path or glob to scan for model ids. Repeatable.",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional revision, tag, or commit SHA to use for every model.",
    )
    parser.add_argument(
        "--cache-dir",
        default=os.environ.get("HF_HUB_CACHE"),
        help="Cache directory passed to snapshot_download. Defaults to HF_HUB_CACHE.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=int(os.environ.get("HF_PREFETCH_WORKERS", "8")),
        help="Parallel workers per repo download. Default: 8.",
    )
    parser.add_argument(
        "--include-framework-extras",
        action="store_true",
        help="Do not skip TF/Flax/ONNX artifacts.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved model list without downloading.",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Continue downloading remaining models if one repo fails.",
    )
    return parser.parse_args()


def is_hf_repo_id(value: str) -> bool:
    value = value.strip()
    if not value:
        return False
    if value.startswith(NON_HF_PREFIXES):
        return False
    if ":" in value:
        return False
    return bool(HF_REPO_RE.match(value))


def unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def expand_config_patterns(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = sorted(PROJECT_ROOT.glob(pattern)) if not Path(pattern).is_absolute() else sorted(Path("/").glob(pattern[1:]))
        if not matches:
            candidate = Path(pattern)
            if not candidate.is_absolute():
                candidate = PROJECT_ROOT / candidate
            matches = [candidate]
        paths.extend(path for path in matches if path.is_file())
    return unique_paths(paths)


def unique_paths(paths: list[Path]) -> list[Path]:
    seen: set[Path] = set()
    result: list[Path] = []
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            result.append(resolved)
    return result


def collect_models_from_configs(config_paths: list[Path]) -> list[str]:
    if yaml is None:
        raise RuntimeError("PyYAML is required for --profile config or --config.")

    found: list[str] = []
    for path in config_paths:
        data = yaml.safe_load(path.read_text()) or {}
        found.extend(collect_models_from_obj(data))
    return unique(found)


def collect_models_from_obj(obj: Any, parent_key: str | None = None) -> list[str]:
    found: list[str] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            key_str = str(key)
            if isinstance(value, str) and key_str in MODEL_KEYS and is_hf_repo_id(value):
                found.append(value)
            else:
                found.extend(collect_models_from_obj(value, key_str))
    elif isinstance(obj, list):
        for item in obj:
            found.extend(collect_models_from_obj(item, parent_key))
    return found


@contextmanager
def repo_lock(repo_id: str, cache_dir: Path | None):
    lock_base = Path(os.environ.get("HF_HOME", "")).expanduser()
    if not str(lock_base):
        lock_base = (cache_dir.parent if cache_dir else Path.home() / ".cache" / "huggingface")
    lock_dir = lock_base / "locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / f"{repo_id.replace('/', '__')}.lock"
    with lock_path.open("w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def resolve_models(args: argparse.Namespace) -> list[str]:
    profiles = args.profile or ["active"]
    models: list[str] = []

    for profile in profiles:
        if profile == "config":
            continue
        models.extend(PROFILE_MODELS[profile])

    config_patterns = list(args.config)
    if "config" in profiles:
        config_patterns.append(ACTIVE_CONFIG_PATTERN)

    if config_patterns:
        config_paths = expand_config_patterns(config_patterns)
        if not config_paths:
            raise RuntimeError(f"No config files matched: {config_patterns}")
        models.extend(collect_models_from_configs(config_paths))

    for model in args.model:
        if not is_hf_repo_id(model):
            raise RuntimeError(f"Not a Hugging Face model repo id: {model}")
        models.append(model)

    return unique(models)


def download_model(repo_id: str, args: argparse.Namespace, cache_dir: Path | None) -> str:
    if snapshot_download is None:
        raise RuntimeError(
            "huggingface_hub is not installed. Activate the dg environment or install "
            "the project requirements first."
        )

    ignore_patterns = None if args.include_framework_extras else list(DEFAULT_IGNORE_PATTERNS)
    kwargs = {
        "repo_id": repo_id,
        "repo_type": "model",
        "revision": args.revision,
        "max_workers": args.max_workers,
        "ignore_patterns": ignore_patterns,
    }
    if cache_dir is not None:
        kwargs["cache_dir"] = str(cache_dir)

    with repo_lock(repo_id, cache_dir):
        return snapshot_download(**kwargs)


def main() -> int:
    args = parse_args()
    cache_dir = Path(args.cache_dir).expanduser() if args.cache_dir else None
    models = resolve_models(args)

    print("Host:", os.uname().nodename)
    print("HF_HOME:", os.environ.get("HF_HOME", "<unset>"))
    print("HF_HUB_CACHE:", os.environ.get("HF_HUB_CACHE", "<unset>"))
    print("Cache dir:", cache_dir or "<huggingface_hub default>")
    print("Models:")
    for model in models:
        print(f"  - {model}")

    if args.dry_run:
        return 0

    failures: list[tuple[str, str]] = []
    for model in models:
        print(f"\n[prefetch] {model}", flush=True)
        try:
            local_path = download_model(model, args, cache_dir)
            print(f"[ok] {model} -> {local_path}", flush=True)
        except Exception as exc:  # noqa: BLE001 - CLI should report all failures.
            message = f"{type(exc).__name__}: {exc}"
            print(f"[failed] {model}: {message}", file=sys.stderr, flush=True)
            failures.append((model, message))
            if not args.keep_going:
                break

    if failures:
        print("\nFailures:", file=sys.stderr)
        for model, message in failures:
            print(f"  - {model}: {message}", file=sys.stderr)
        return 1

    print("\nAll requested models are cached.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
