"""Every task appends exactly one structured JSONL record — Phase 2's training set."""

import json

from router.task_log import TaskLogger


def test_append_writes_one_json_line_per_record(tmp_path):
    path = tmp_path / "log.jsonl"
    logger = TaskLogger(path)
    logger.append({"task_id": "t1", "domain": "math_reasoning", "stage": "stage1_tool",
                   "answer": "4", "verified": True, "cost_usd": 0.0})
    logger.append({"task_id": "t2", "domain": "factual_knowledge", "stage": "stage3_paid",
                   "answer": "Paris", "verified": False, "cost_usd": 0.0001})

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    rec1 = json.loads(lines[0])
    assert rec1["task_id"] == "t1"
    assert rec1["verified"] is True
    assert json.loads(lines[1])["cost_usd"] == 0.0001


def test_append_requires_task_id(tmp_path):
    logger = TaskLogger(tmp_path / "log.jsonl")
    try:
        logger.append({"domain": "ner"})
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_non_ascii_preserved(tmp_path):
    path = tmp_path / "log.jsonl"
    TaskLogger(path).append({"task_id": "t1", "answer": "café ≥ 3"})
    assert json.loads(path.read_text(encoding="utf-8"))["answer"] == "café ≥ 3"
