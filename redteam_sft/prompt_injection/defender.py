"""Defender message construction."""

from __future__ import annotations

from typing import Any


def build_defender_messages(
    system_prompt_template: str,
    case: dict[str, Any],
    payload: str,
) -> list[dict[str, str]]:
    secret = case.get("secret", "")
    system_prompt = system_prompt_template.replace("{{SECRET_KEY}}", secret)

    if case.get("defender_user_message"):
        user_content = case["defender_user_message"].replace(
            "{{ATTACK_PAYLOAD}}", payload,
        )
    else:
        pieces = []
        if case.get("defender_user_task"):
            pieces.append(case["defender_user_task"].strip())
        if case.get("carrier_context"):
            pieces.append(case["carrier_context"].strip())
        pieces.append(payload.strip())
        user_content = "\n\n".join(piece for piece in pieces if piece)

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
