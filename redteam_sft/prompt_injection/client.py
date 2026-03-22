"""OpenAI-compatible HTTP chat client used for both defender and judge."""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


class OpenAICompatibleChatClient:
    def __init__(self, client_cfg: dict[str, Any]):
        self.base_url = (client_cfg.get("base_url") or "").rstrip("/")
        self.api_key_env = client_cfg.get("api_key_env")
        self.api_key = ""
        if self.api_key_env:
            self.api_key = os.getenv(self.api_key_env, "")
            if not self.api_key:
                logger.warning(
                    "Environment variable %s is not set; "
                    "requests will be sent without authentication.",
                    self.api_key_env,
                )
        self.timeout = client_cfg.get("timeout_seconds", 120)
        self.model_id = client_cfg["model_id"]
        self.default_generation = client_cfg.get("generation", {})
        if not self.base_url:
            raise ValueError("OpenAI-compatible client requires base_url")

    def create_chat_completion(
        self,
        messages: list[dict[str, str]],
        generation: dict[str, Any] | None = None,
    ) -> str:
        import requests

        url = self.base_url + "/chat/completions"
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        merged_generation = dict(self.default_generation)
        if generation:
            merged_generation.update(generation)

        payload: dict[str, Any] = {
            "model": self.model_id,
            "messages": messages,
            "temperature": merged_generation.get("temperature", 0.0),
            "top_p": merged_generation.get("top_p", 1.0),
            "max_tokens": merged_generation.get("max_new_tokens", 256),
        }
        if "seed" in merged_generation:
            payload["seed"] = merged_generation["seed"]

        response = requests.post(
            url, headers=headers, json=payload, timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()
