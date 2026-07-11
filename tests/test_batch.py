"""Batch job: JSONL/CSV dataset in, results JSONL out."""

import json

from router.batch import read_tasks, run_batch


def test_read_tasks_jsonl(tmp_path):
    p = tmp_path / "in.jsonl"
    p.write_text(
        '{"id": "a", "prompt": "2+2?"}\n'
        '{"task_id": "b", "question": "capital?", "tests": "assert True"}\n',
        encoding="utf-8",
    )
    tasks = read_tasks(p)
    assert [t.task_id for t in tasks] == ["a", "b"]
    assert tasks[0].prompt == "2+2?"
    assert tasks[1].prompt == "capital?"
    assert tasks[1].metadata["tests"] == "assert True"


def test_read_tasks_csv(tmp_path):
    p = tmp_path / "in.csv"
    p.write_text('id,prompt,expected\na,"2+2?",4\n', encoding="utf-8")
    tasks = read_tasks(p)
    assert tasks[0].task_id == "a"
    assert tasks[0].prompt == "2+2?"
    assert tasks[0].metadata["expected"] == "4"


def test_read_tasks_generates_ids_when_missing(tmp_path):
    p = tmp_path / "in.jsonl"
    p.write_text('{"prompt": "one"}\n{"prompt": "two"}\n', encoding="utf-8")
    tasks = read_tasks(p)
    assert len({t.task_id for t in tasks}) == 2


def test_run_batch_writes_one_result_per_task(tmp_path):
    class FakePipeline:
        def process(self, task):
            return {"task_id": task.task_id, "answer": task.prompt.upper(),
                    "cost_usd": 0.001, "verified": True, "stage": "stage1_tool"}

    inp = tmp_path / "in.jsonl"
    inp.write_text('{"id": "a", "prompt": "x"}\n{"id": "b", "prompt": "y"}\n', encoding="utf-8")
    out = tmp_path / "out.jsonl"
    summary = run_batch(FakePipeline(), inp, out)
    lines = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert [r["answer"] for r in lines] == ["X", "Y"]
    assert summary["tasks"] == 2
    assert summary["total_cost_usd"] == 0.002


def test_run_batch_task_failure_yields_empty_answer_not_crash(tmp_path):
    class FlakyPipeline:
        def process(self, task):
            if task.task_id == "a":
                raise RuntimeError("boom")
            return {"task_id": task.task_id, "answer": "ok", "cost_usd": 0.0}

    inp = tmp_path / "in.jsonl"
    inp.write_text('{"id": "a", "prompt": "x"}\n{"id": "b", "prompt": "y"}\n', encoding="utf-8")
    out = tmp_path / "out.jsonl"
    summary = run_batch(FlakyPipeline(), inp, out)
    lines = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 2
    assert lines[0]["answer"] == ""
    assert "boom" in lines[0]["error"]
    assert summary["errors"] == 1
