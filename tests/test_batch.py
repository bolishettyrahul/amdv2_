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


def test_read_tasks_json_array_pretty_printed(tmp_path):
    p = tmp_path / "tasks.json"
    p.write_text(
        json.dumps([{"id": "a", "prompt": "2+2?"},
                    {"task_id": "b", "question": "capital?", "tests": "assert True"}],
                   indent=2),
        encoding="utf-8",
    )
    tasks = read_tasks(p)
    assert [t.task_id for t in tasks] == ["a", "b"]
    assert tasks[1].metadata["tests"] == "assert True"


def test_read_tasks_json_array_single_line(tmp_path):
    p = tmp_path / "tasks.json"
    p.write_text('[{"id": "a", "prompt": "x"}, {"id": "b", "prompt": "y"}]', encoding="utf-8")
    tasks = read_tasks(p)
    assert [t.task_id for t in tasks] == ["a", "b"]


def test_read_tasks_json_object_wrapping_task_list(tmp_path):
    p = tmp_path / "tasks.json"
    p.write_text('{"tasks": [{"id": "a", "prompt": "x"}]}', encoding="utf-8")
    tasks = read_tasks(p)
    assert [t.task_id for t in tasks] == ["a"]
    assert tasks[0].prompt == "x"


def test_run_batch_json_output_is_single_valid_json_array(tmp_path):
    class FakePipeline:
        def process(self, task):
            if task.task_id == "b":
                raise RuntimeError("boom")
            return {"task_id": task.task_id, "answer": "ok", "cost_usd": 0.001}

    inp = tmp_path / "tasks.json"
    inp.write_text('[{"id": "a", "prompt": "x"}, {"id": "b", "prompt": "y"}]', encoding="utf-8")
    out = tmp_path / "results.json"
    summary = run_batch(FakePipeline(), inp, out)
    results = json.loads(out.read_text(encoding="utf-8"))  # whole file must be valid JSON
    assert isinstance(results, list)
    assert [r["task_id"] for r in results] == ["a", "b"]
    assert results[1]["answer"] == ""  # failed task still gets a result entry
    assert summary["tasks"] == 2


def test_harness_json_output_carries_only_task_id_and_string_answer(tmp_path):
    class FakePipeline:
        def process(self, task):
            return {"task_id": task.task_id, "answer": 42,  # non-string answer
                    "cost_usd": 0.001, "verified": True, "stage": "stage1_tool"}

    inp = tmp_path / "tasks.json"
    inp.write_text('[{"id": "a", "prompt": "x"}]', encoding="utf-8")
    out = tmp_path / "results.json"
    run_batch(FakePipeline(), inp, out)
    results = json.loads(out.read_text(encoding="utf-8"))
    # Exactly the reference-submission schema: nothing but task_id + answer.
    assert results == [{"task_id": "a", "answer": "42"}]


def test_exhausted_budget_still_writes_every_task_id(tmp_path):
    class MustNotRun:
        def process(self, task):
            raise AssertionError("dispatched a task after the budget expired")

    inp = tmp_path / "tasks.json"
    inp.write_text('[{"id": "a", "prompt": "x"}, {"id": "b", "prompt": "y"}]', encoding="utf-8")
    out = tmp_path / "results.json"
    summary = run_batch(MustNotRun(), inp, out, budget_s=-1.0)
    results = json.loads(out.read_text(encoding="utf-8"))
    assert [r["task_id"] for r in results] == ["a", "b"]
    assert all(r["answer"] == "" for r in results)
    assert summary["tasks"] == 2


def test_fatal_interrupt_still_leaves_complete_valid_output(tmp_path):
    import pytest

    class InterruptedPipeline:
        def process(self, task):
            if task.task_id == "b":
                raise KeyboardInterrupt  # harness kill mid-run
            return {"task_id": task.task_id, "answer": "ok", "cost_usd": 0.0}

    inp = tmp_path / "tasks.json"
    inp.write_text(
        '[{"id": "a", "prompt": "x"}, {"id": "b", "prompt": "y"}, {"id": "c", "prompt": "z"}]',
        encoding="utf-8")
    out = tmp_path / "results.json"
    with pytest.raises(KeyboardInterrupt):
        run_batch(InterruptedPipeline(), inp, out)
    results = json.loads(out.read_text(encoding="utf-8"))
    assert [r["task_id"] for r in results] == ["a", "b", "c"]
    assert results[0]["answer"] == "ok"
    assert results[1]["answer"] == results[2]["answer"] == ""


def test_unreadable_input_still_writes_valid_empty_json(tmp_path):
    out = tmp_path / "results.json"
    summary = run_batch(None, tmp_path / "nope.json", out)
    assert json.loads(out.read_text(encoding="utf-8")) == []
    assert summary["tasks"] == 0


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
