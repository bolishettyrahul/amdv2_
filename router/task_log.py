"""Append-only JSONL dataset log — one record per task, Phase 2's training set."""

from __future__ import annotations

import json
from pathlib import Path


class TaskLogger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: dict) -> None:
        if "task_id" not in record:
            raise ValueError("log record must include task_id")
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
