"""Append-only JSONL memory store with lazy byte-offset indexing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator, Optional


class JsonlMemoryStore:
    """Simple append-only JSONL store keyed by one field.

    The offset index is built lazily on first keyed lookup. When duplicate keys
    exist, the latest record wins because its byte offset overwrites the older
    one in the in-memory index.
    """

    def __init__(
        self,
        path: str | Path,
        key_field: str = "episode_key",
    ) -> None:
        self.path = Path(path)
        self.key_field = key_field
        self._offsets: dict[str, int] = {}
        self._index_built = False

    def append(self, record: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        key = record.get(self.key_field)

        with open(self.path, "a", encoding="utf-8") as f:
            offset = f.tell()
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        if self._index_built and key:
            self._offsets[str(key)] = offset

    def contains(self, key: str) -> bool:
        self._ensure_index()
        return str(key) in self._offsets

    def get(self, key: str) -> Optional[dict[str, Any]]:
        self._ensure_index()
        offset = self._offsets.get(str(key))
        if offset is None or not self.path.exists():
            return None

        with open(self.path, "r", encoding="utf-8") as f:
            f.seek(offset)
            line = f.readline()
        return self._parse_line(line)

    def iter_records(self) -> Iterator[dict[str, Any]]:
        if not self.path.exists():
            return

        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                row = self._parse_line(line)
                if row is not None:
                    yield row

    def __len__(self) -> int:
        self._ensure_index()
        return len(self._offsets)

    def _ensure_index(self) -> None:
        if self._index_built:
            return

        self._offsets = {}
        if not self.path.exists():
            self._index_built = True
            return

        with open(self.path, "r", encoding="utf-8") as f:
            while True:
                offset = f.tell()
                line = f.readline()
                if not line:
                    break
                row = self._parse_line(line)
                if row is None:
                    continue
                key = row.get(self.key_field)
                if key:
                    self._offsets[str(key)] = offset

        self._index_built = True

    @staticmethod
    def _parse_line(line: str) -> Optional[dict[str, Any]]:
        line = line.lstrip("\ufeff").strip()
        if not line:
            return None
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            return None
