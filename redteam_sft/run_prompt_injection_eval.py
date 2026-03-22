#!/usr/bin/env python3
"""
run_prompt_injection_eval.py

Prompt injection evaluation harness entry point.

Usage:
    python redteam_sft/run_prompt_injection_eval.py \
        --config redteam_sft/prompt_injection_eval_kda_vs_sft.yaml
"""

import sys

from redteam_sft.prompt_injection import run

if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        sys.exit(130)
