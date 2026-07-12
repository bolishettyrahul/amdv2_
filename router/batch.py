"""Batch job I/O: dataset file (JSONL/CSV) in, results JSONL out."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from router.types import Task

_ID_KEYS = ("task_id", "id")
_PROMPT_KEYS = ("prompt", "question", "input", "text")


def _to_task(row: dict, index: int) -> Task:
    task_id = next((str(row[k]) for k in _ID_KEYS if row.get(k) not in (None, "")), f"task-{index}")
    prompt = next((str(row[k]) for k in _PROMPT_KEYS if row.get(k) not in (None, "")), "")
    consumed = set(_ID_KEYS) | set(_PROMPT_KEYS)
    metadata = {k: v for k, v in row.items() if k not in consumed and v not in (None, "")}
    return Task(task_id=task_id, prompt=prompt, metadata=metadata)


def read_tasks(path: str | Path) -> list[Task]:
    path = Path(path)
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
    else:  # JSONL
        with path.open(encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
    return [_to_task(row, i) for i, row in enumerate(rows)]


def run_batch(pipeline, in_path: str | Path, out_path: str | Path) -> dict:
    tasks = read_tasks(in_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    total_cost = 0.0
    errors = 0
    with out_path.open("w", encoding="utf-8") as out:
        for task in tasks:
            try:
                record = pipeline.process(task)
                result = {"task_id": task.task_id, "answer": record.get("answer") or "",
                          **{k: record[k] for k in ("stage", "verified", "cost_usd",
                                                    "allowed_models_fallback")
                             if k in record}}
                total_cost += record.get("cost_usd", 0.0)
            except Exception as exc:  # one bad task must not sink the whole run
                errors += 1
                result = {"task_id": task.task_id, "answer": "", "error": str(exc)}
            out.write(json.dumps(result, ensure_ascii=False, default=str) + "\n")
    return {"tasks": len(tasks), "errors": errors, "total_cost_usd": total_cost}
