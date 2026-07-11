"""Core domain/task types shared across the cascade."""

from router.types import Domain, Task


def test_eight_domains():
    assert {d.value for d in Domain} == {
        "factual_knowledge",
        "math_reasoning",
        "sentiment_classification",
        "summarization",
        "ner",
        "code_debugging",
        "logical_reasoning",
        "code_generation",
    }


def test_task_carries_id_prompt_and_metadata():
    t = Task(task_id="t1", prompt="What is 2+2?")
    assert t.task_id == "t1"
    assert t.prompt == "What is 2+2?"
    assert t.metadata == {}
