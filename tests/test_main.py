"""The CLI entrypoint: parse args, wire the pipeline, run the batch, print the summary.

These tests inject a fake settings loader and pipeline builder so the real
factory.build_pipeline (startup health check + sentence-transformers load) never
runs — proving main() does no network I/O of its own beyond what the pipeline does.
"""

import json

from router import main as main_mod
from router.config import Settings
from router.types import Task


class StubPipeline:
    """Stands in for a real Pipeline: records processed tasks, returns fixed records."""

    def __init__(self, record_for):
        self.record_for = record_for
        self.processed = []

    def process(self, task: Task) -> dict:
        self.processed.append(task)
        return self.record_for(task)


def write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def test_main_wires_settings_into_builder_and_runs_batch(tmp_path, capsys):
    in_path = tmp_path / "tasks.jsonl"
    out_path = tmp_path / "results.jsonl"
    write_jsonl(in_path, [
        {"task_id": "f1", "prompt": "Capital of France?"},
        {"task_id": "m1", "prompt": "What is 2+2?"},
        {"task_id": "c1", "prompt": "Fix this bug"},
    ])

    injected = Settings(paid_provider="groq", groq_api_key="test-key")
    seen = {}

    def fake_load_settings():
        return injected

    def fake_builder(settings):
        seen["settings"] = settings
        return StubPipeline(lambda t: {"answer": f"ans-{t.task_id}", "stage": "stage1_tool",
                                       "verified": True, "cost_usd": 0.0})

    summary = main_mod.main(
        ["--input", str(in_path), "--output", str(out_path)],
        load_settings=fake_load_settings,
        pipeline_builder=fake_builder,
    )

    # The real factory was never called; our injected settings reached the builder.
    assert seen["settings"] is injected
    assert summary == {"tasks": 3, "errors": 0, "total_cost_usd": 0.0}

    lines = out_path.read_text(encoding="utf-8").splitlines()
    assert [json.loads(l)["answer"] for l in lines] == ["ans-f1", "ans-m1", "ans-c1"]

    # The human running it sees the summary numbers on stdout.
    out = capsys.readouterr().out
    assert "3" in out
    assert "0" in out


def test_main_sums_cost_and_counts_errors_from_the_batch(tmp_path, capsys):
    in_path = tmp_path / "tasks.jsonl"
    out_path = tmp_path / "results.jsonl"
    write_jsonl(in_path, [{"task_id": "a"}, {"task_id": "b"}])

    def record_for(task):
        if task.task_id == "b":
            raise RuntimeError("stage blew up")
        return {"answer": "ok", "stage": "stage3_paid", "verified": False, "cost_usd": 0.0025}

    summary = main_mod.main(
        ["--input", str(in_path), "--output", str(out_path)],
        load_settings=lambda: Settings(),
        pipeline_builder=lambda s: StubPipeline(record_for),
    )

    assert summary["tasks"] == 2
    assert summary["errors"] == 1
    assert summary["total_cost_usd"] == 0.0025
    out = capsys.readouterr().out
    assert "0.0025" in out or "0.002" in out


def test_zero_args_default_to_harness_mounts():
    # The grading harness runs the container with no CLI arguments.
    args = main_mod._parse_args([])
    assert args.input == "/input/tasks.json"
    assert args.output == "/output/results.json"


def test_cli_paths_remain_overridable_for_local_runs():
    args = main_mod._parse_args(["--input", "fixtures/sample_tasks.jsonl",
                                 "--output", "out.jsonl"])
    assert args.input == "fixtures/sample_tasks.jsonl"
    assert args.output == "out.jsonl"
