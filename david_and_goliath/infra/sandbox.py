"""Best-effort Python subprocess isolation for Blue Team verification.

This is not a hard security boundary. It provides:
  - isolated temp working directory
  - `python -I -B` child interpreter
  - minimal inherited environment
  - timeout limits
  - denylist-based safety screening before execution

It is intentionally lightweight and Windows-friendly.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

SAFE_ENV_KEYS = (
    "SYSTEMROOT",
    "WINDIR",
    "COMSPEC",
    "PATHEXT",
    "OS",
    "TEMP",
    "TMP",
    "HOME",
    "USERPROFILE",
    "LANG",
    "LC_ALL",
)

_DANGEROUS_PATTERNS: list[tuple[str, str]] = [
    ("network", r"(?i)\b(requests|urllib|http\.client|socket|aiohttp)\b"),
    ("shell", r"(?i)\bsubprocess\b|os\.(system|popen)\b"),
    ("dynamic_eval", r"(?i)\b(eval|exec|compile)\s*\("),
    ("env_access", r"(?i)os\.(environ|getenv)\b"),
    ("dynamic_import", r"(?i)__import__\s*\("),
    ("ctypes", r"(?i)\bctypes\b"),
    ("absolute_open", r"open\s*\(\s*['\"]([A-Za-z]:[\\/]|/)"),
]


def _build_child_env() -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k.upper() in SAFE_ENV_KEYS}
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    return env


def screen_python_code(code: str) -> dict[str, Any]:
    matches: list[dict[str, str]] = []
    for rule_id, pattern in _DANGEROUS_PATTERNS:
        hit = re.search(pattern, code)
        if hit:
            matches.append(
                {
                    "rule": rule_id,
                    "matched_text": hit.group(0)[:120],
                }
            )

    return {
        "blocked": bool(matches),
        "match_count": len(matches),
        "matches": matches,
        "summary": (
            "Execution blocked by safety screen: "
            + ", ".join(match["rule"] for match in matches)
            if matches
            else "No blocked execution patterns detected."
        ),
    }


def run_python_in_sandbox(
    code: str,
    timeout: int,
    *,
    label: str = "snippet",
) -> dict[str, Any]:
    """Run Python code in an isolated temp dir with minimal environment."""
    screening = screen_python_code(code)
    if screening["blocked"]:
        return {
            "stdout": "",
            "stderr": screening["summary"],
            "returncode": -1,
            "elapsed_ms": 0.0,
            "timed_out": False,
            "blocked": True,
            "success": False,
            "screening": screening,
        }

    sandbox_root = Path.cwd() / ".dg_sandbox"
    sandbox_root.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".py",
        prefix=f"{label}_",
        dir=sandbox_root,
        encoding="utf-8",
        delete=False,
    ) as fh:
        fh.write(code)
        script_path = Path(fh.name)

    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            [sys.executable, "-I", "-B", str(script_path)],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            env=_build_child_env(),
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000
        return {
            "stdout": proc.stdout[:2000],
            "stderr": proc.stderr[:1000],
            "returncode": proc.returncode,
            "elapsed_ms": round(elapsed_ms, 1),
            "timed_out": False,
            "blocked": False,
            "success": proc.returncode == 0,
            "screening": screening,
        }
    except subprocess.TimeoutExpired:
        return {
            "stdout": "",
            "stderr": f"Execution timed out after {timeout}s.",
            "returncode": -1,
            "elapsed_ms": timeout * 1000,
            "timed_out": True,
            "blocked": False,
            "success": False,
            "screening": screening,
        }
    except Exception as exc:
        return {
            "stdout": "",
            "stderr": str(exc),
            "returncode": -1,
            "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1),
            "timed_out": False,
            "blocked": False,
            "success": False,
            "screening": screening,
        }
    finally:
        script_path.unlink(missing_ok=True)
