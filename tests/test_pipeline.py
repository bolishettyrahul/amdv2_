"""The cascade orchestrator: stage gates, early $0 returns, logging, critique."""

import json

from router.critique import CritiqueResult
from router.pipeline import Pipeline
from router.task_log import TaskLogger
from router.types import Domain, StageResult, Task


class StubClassifier:
    def __init__(self, domain):
        self.domain = domain

    def classify(self, prompt):
        return self.domain, 0.9


class StubStage:
    def __init__(self, result):
        self.result = result
        self.calls = 0

    def attempt(self, task, domain):
        self.calls += 1
        return self.result


class StubCritic:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.calls = []

    def critique(self, task, domain, answer, actor_model):
        self.calls.append((domain, answer, actor_model))
        if self.error:
            raise self.error
        return self.result


def s(answer, stage, verified, model=None, cost=0.0):
    return StageResult(answer, stage, verified=verified, model=model, cost_usd=cost)


def make_pipeline(tmp_path, domain, s1, s2, s3, critic=None):
    return Pipeline(
        classifier=StubClassifier(domain),
        stage1=s1, stage2=s2, stage3=s3,
        critic=critic or StubCritic(),
        logger=TaskLogger(tmp_path / "log.jsonl"),
    )


def read_log(tmp_path):
    lines = (tmp_path / "log.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


def test_stage1_verified_short_circuits(tmp_path):
    s1 = StubStage(s("4", "stage1_tool", True))
    s2 = StubStage(s("x", "stage2_local", True))
    s3 = StubStage(s("x", "stage3_paid", True))
    p = make_pipeline(tmp_path, Domain.MATH, s1, s2, s3)
    record = p.process(Task("t1", "2+2?"))
    assert record["answer"] == "4"
    assert record["stage"] == "stage1_tool"
    assert record["cost_usd"] == 0.0
    assert (s2.calls, s3.calls) == (0, 0)
    [logged] = read_log(tmp_path)
    assert logged["task_id"] == "t1"
    assert logged["domain"] == "math_reasoning"


def test_cascade_falls_through_to_stage3(tmp_path):
    s1 = StubStage(s(None, "stage1_tool", False))
    s2 = StubStage(s("bad", "stage2_local", False, model="ollama/qwen2.5:7b"))
    s3 = StubStage(s("good", "stage3_paid", False,
                     model="accounts/fireworks/models/gemma2-9b-it", cost=0.001))
    p = make_pipeline(tmp_path, Domain.FACTUAL, s1, s2, s3)
    record = p.process(Task("t1", "hard question"))
    assert record["answer"] == "good"
    assert record["stage"] == "stage3_paid"
    assert record["cost_usd"] == 0.001
    assert (s1.calls, s2.calls, s3.calls) == (1, 1, 1)


def test_critique_runs_for_weak_domain_and_is_logged(tmp_path):
    critic = StubCritic(CritiqueResult("correct", "looks right", "critic-model"))
    s2 = StubStage(s("Canberra", "stage2_local", True, model="ollama/qwen2.5:7b"))
    p = make_pipeline(tmp_path, Domain.FACTUAL,
                      StubStage(s(None, "stage1_tool", False)), s2,
                      StubStage(s("x", "stage3_paid", True)), critic)
    record = p.process(Task("t1", "Capital of Australia?"))
    assert critic.calls == [(Domain.FACTUAL, "Canberra", "ollama/qwen2.5:7b")]
    assert record["critique"]["verdict"] == "correct"
    [logged] = read_log(tmp_path)
    assert logged["critique"]["verdict"] == "correct"


def test_critique_failure_never_blocks_the_answer(tmp_path):
    critic = StubCritic(error=RuntimeError("all critics down"))
    s2 = StubStage(s("Canberra", "stage2_local", True, model="ollama/qwen2.5:7b"))
    p = make_pipeline(tmp_path, Domain.FACTUAL,
                      StubStage(s(None, "stage1_tool", False)), s2,
                      StubStage(s("x", "stage3_paid", True)), critic)
    record = p.process(Task("t1", "q"))
    assert record["answer"] == "Canberra"
    assert "error" in record["critique"]


def test_no_stage2_in_cloud_fallback_wiring_stage3_still_reached(tmp_path):
    # Pipeline works when stage2 is replaced (cloud fallback) or identical API.
    s3 = StubStage(s("ans", "stage3_paid", True, cost=0.002))
    p = make_pipeline(tmp_path, Domain.SUMMARIZATION,
                      StubStage(s(None, "stage1_tool", False)),
                      StubStage(s(None, "stage2_local", False)), s3)
    record = p.process(Task("t1", "Summarize ..."))
    assert record["stage"] == "stage3_paid"


def test_stage_error_escalates_instead_of_crashing(tmp_path):
    class BoomStage:
        def attempt(self, task, domain):
            raise RuntimeError("ollama went away")

    s3 = StubStage(s("recovered", "stage3_paid", True))
    p = make_pipeline(tmp_path, Domain.FACTUAL,
                      StubStage(s(None, "stage1_tool", False)), BoomStage(), s3)
    record = p.process(Task("t1", "q"))
    assert record["answer"] == "recovered"
    assert record["stage"] == "stage3_paid"
