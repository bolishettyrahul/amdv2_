"""Batch job I/O: dataset file (JSONL/CSV) in, results JSONL out."""

from __future__ import annotations

import csv
import json
import os
import sys
import time
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


def _parse_json_rows(text: str) -> list[dict]:
    """Rows from a JSON document: a bare array, or an object wrapping one."""
    doc = json.loads(text)
    if isinstance(doc, list):
        return doc
    if isinstance(doc, dict):
        if any(k in doc for k in _ID_KEYS + _PROMPT_KEYS):
            return [doc]  # single task object
        for value in doc.values():
            if isinstance(value, list) and all(isinstance(r, dict) for r in value):
                return value
        return [doc]
    raise ValueError(f"unsupported JSON document of type {type(doc).__name__}")


def read_tasks(path: str | Path) -> list[Task]:
    path = Path(path)
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
    else:
        # The grading harness ships tasks.json (one JSON array); local dev uses
        # JSONL. Sniff the content rather than trusting the extension.
        text = path.read_text(encoding="utf-8-sig")
        try:
            # Whole file as one JSON document (array, wrapper object, or task).
            rows = _parse_json_rows(text)
        except json.JSONDecodeError:
            # Multiple documents -> JSONL, one object per line.
            rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    return [_to_task(row, i) for i, row in enumerate(rows)]


def _write_results(out_path: Path, results: list[dict]) -> None:
    """Atomic write: a partial results file is worse than the previous one.

    Harness mode (`.json`) strips every record to the reference-submission
    schema — exactly task_id + a string answer, as one valid JSON document.
    Local dev (`.jsonl`) keeps the full per-task record, one per line.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as out:
        if out_path.suffix.lower() == ".json":
            payload = [{"task_id": str(r["task_id"]),
                        "answer": r["answer"] if isinstance(r["answer"], str)
                        else "" if r["answer"] is None else str(r["answer"])}
                       for r in results]
            json.dump(payload, out, ensure_ascii=False)
        else:
            for result in results:
                out.write(json.dumps(result, ensure_ascii=False, default=str) + "\n")
    tmp.replace(out_path)


def run_batch(pipeline, in_path: str | Path, out_path: str | Path,
              budget_s: float | None = None) -> dict:
    """Answer every task, honouring two hard harness rules: every input task_id
    appears exactly once in the output, and the output file is valid on exit
    even if the run is interrupted or the time budget expires."""
    if budget_s is None:
        # The harness kills the container at 10 minutes; stop dispatching with
        # enough slack left to serialise what we have.
        budget_s = float(os.environ.get("RUN_BUDGET_S", str(9 * 60 + 15)))
    out_path = Path(out_path)
    try:
        tasks = read_tasks(in_path)
    except Exception as exc:  # no tasks -> nothing valid to write beyond []
        print(f"could not read {in_path}: {exc!r}", file=sys.stderr)
        _write_results(out_path, [])
        return {"tasks": 0, "errors": 1, "total_cost_usd": 0.0}

    # Seed every task_id up front; the loop only ever improves an answer.
    results: list[dict] = [{"task_id": t.task_id, "answer": ""} for t in tasks]
    total_cost = 0.0
    errors = 0
    started = time.monotonic()
    try:
        for i, task in enumerate(tasks):
            elapsed = time.monotonic() - started
            if elapsed > budget_s:
                print(f"run budget exhausted after {i}/{len(tasks)} tasks "
                      f"({elapsed:.0f}s); remaining tasks keep the empty answer",
                      file=sys.stderr)
                break
            try:
                record = pipeline.process(task)
                results[i] = {"task_id": task.task_id, "answer": record.get("answer") or "",
                              **{k: record[k] for k in ("stage", "verified", "cost_usd",
                                                        "allowed_models_fallback")
                                 if k in record}}
                total_cost += record.get("cost_usd", 0.0)
            except Exception as exc:  # one bad task must not sink the whole run
                errors += 1
                results[i] = {"task_id": task.task_id, "answer": "", "error": str(exc)}
    finally:
        # Runs on normal return, an unexpected raise, and the harness-kill
        # (KeyboardInterrupt/SystemExit) paths.
        _write_results(out_path, results)
    return {"tasks": len(tasks), "errors": errors, "total_cost_usd": total_cost}
