"""Offline evaluation CLI for the Phase 1 cascade.

Runs fixtures/sample_tasks.jsonl through a real Pipeline (router.factory.build_pipeline),
scores it against fixtures/expected.jsonl with router.scoring.score_run, then sweeps the
verification-gate parameters named in plan/evaluation-tuning-strategy.md (sentiment_threshold,
factual_k, logic_k) plus sandbox_timeout_s, rebuilding a Pipeline per configuration. Reports a
table sorted to surface the Pareto frontier (accuracy vs. total paid cost).

Without Fireworks/Groq credentials in the environment, this runs against a built-in offline
transport that fakes an LLM by echoing/deriving the *expected* answer per task -- enough to
exercise every deterministic verifier (sympy, VADER, z3, sandboxed test execution) and the
weak-verifier gates (self-consistency, NER schema, summary heuristic) for real. Pass a real
FIREWORKS_API_KEY/GROQ_API_KEY to hit a live provider instead.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from router.batch import read_tasks  # noqa: E402
from router.config import Settings  # noqa: E402
from router.factory import build_pipeline  # noqa: E402
from router.scoring import score_run  # noqa: E402
from router.tools.sentiment_tool import extract_target_text  # noqa: E402
from router.types import Task  # noqa: E402
from router.verifiers import summary_ok  # noqa: E402

# Sweep ranges for the parameters plan/evaluation-tuning-strategy.md names as the
# calibration knobs, mapped onto the matching router.config.Settings fields
# (sentiment_confidence_threshold -> sentiment_threshold, factual_self_consistency_k ->
# factual_k, logical_self_consistency_k -> logic_k), using the doc's own "Sweep Range"
# column. sandbox_timeout_s is not in the doc's table (that table covers
# code_retry_limit instead) -- its range below is chosen around the "~2s per test
# execution" target in CLAUDE.md, spanning tight to generous.
SWEEP_RANGES: dict[str, list] = {
    "sentiment_threshold": [0.4, 0.5, 0.65, 0.8, 0.9],
    "factual_k": [1, 2, 3, 4, 5],
    "logic_k": [1, 2, 3, 4, 5],
    "sandbox_timeout_s": [0.5, 1.0, 2.0, 4.0],
}

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def load_fixtures(fixtures_dir: Path) -> tuple[list[Task], list[str]]:
    tasks = read_tasks(fixtures_dir / "sample_tasks.jsonl")
    with (fixtures_dir / "expected.jsonl").open(encoding="utf-8") as f:
        expected = [json.loads(line) for line in f if line.strip()]
    if len(tasks) != len(expected):
        raise ValueError(
            f"sample_tasks.jsonl has {len(tasks)} rows but expected.jsonl has {len(expected)}"
        )
    return tasks, expected


def _extractive_summary(source: str) -> str:
    """Leading-sentence summary, grown until it clears summary_ok's heuristic."""
    sentences = [s.strip() for s in _SENTENCE_SPLIT.split(source.strip()) if s.strip()]
    for k in range(1, len(sentences) + 1):
        candidate = " ".join(sentences[:k])
        ok, _ = summary_ok(source, candidate)
        if ok:
            return candidate
    return " ".join(sentences[:2]) or source


def _chat_body(text: str) -> dict:
    return {"choices": [{"message": {"role": "assistant", "content": text}}]}


def build_fake_transport(tasks: list[Task], expected: list[str]):
    """An offline stand-in LLM: derives a plausible completion per task's domain so every
    real verifier (sympy/VADER/z3/sandbox, self-consistency, NER schema, summary heuristic)
    still runs against genuine input, without needing network access or paid credentials.
    """
    expected_by_id = {t.task_id: e for t, e in zip(tasks, expected)}
    # Longest-prompt-first so no shorter prompt can accidentally prefix-match first.
    by_prompt = sorted(tasks, key=lambda t: -len(t.prompt))

    def find_task(content: str) -> Task | None:
        for task in by_prompt:
            if content.startswith(task.prompt):
                return task
        return None

    def transport(url: str, headers: dict, payload: dict) -> tuple[int, dict]:
        messages = payload.get("messages", [])
        if len(messages) == 1 and messages[0]["content"].startswith(
            "You are grading another model's answer."
        ):
            return 200, _chat_body("VERDICT: correct\nREASONING: matches the expected answer.")

        content = messages[1]["content"] if len(messages) > 1 else (
            messages[-1]["content"] if messages else ""
        )
        task = find_task(content)
        if task is None:
            return 200, _chat_body("")

        expected_answer = expected_by_id[task.task_id]
        domain = task.metadata.get("domain")
        if task.metadata.get("solution"):
            text = f"```python\n{task.metadata['solution']}\n```"
        elif domain == "ner":
            text = json.dumps([{"text": expected_answer, "label": "ENTITY"}])
        elif domain == "summarization":
            text = _extractive_summary(extract_target_text(task.prompt))
        else:
            # factual / logic / math / sentiment fallback: the free verifiers for these
            # (self-consistency, sympy re-evaluation, schema check) just need a stable,
            # correct-looking answer -- echoing the label directly satisfies them.
            text = expected_answer
        return 200, _chat_body(text)

    return transport


def run_config(tasks: list[Task], expected: list[str], settings: Settings, transport,
                log_dir: Path, label: str) -> dict:
    settings = replace(settings, log_path=str(log_dir / f"{label}.jsonl"))
    pipeline = build_pipeline(settings, transport=transport)
    records = [pipeline.process(t) for t in tasks]
    result = score_run(records, expected)
    result["label"] = label
    return result


def pareto_frontier(results: list[dict]) -> set[str]:
    frontier = set()
    for r in results:
        dominated = any(
            o is not r
            and o["accuracy"] >= r["accuracy"]
            and o["total_cost_usd"] <= r["total_cost_usd"]
            and (o["accuracy"] > r["accuracy"] or o["total_cost_usd"] < r["total_cost_usd"])
            for o in results
        )
        if not dominated:
            frontier.add(r["label"])
    return frontier


def print_table(results: list[dict]) -> None:
    frontier = pareto_frontier(results)
    ordered = sorted(results, key=lambda r: (r["total_cost_usd"], -r["accuracy"]))
    header = f"{'config':28s} {'accuracy':>9s} {'correct':>9s} {'cost_usd':>12s} {'score':>14s}  pareto"
    print(header)
    print("-" * len(header))
    for r in ordered:
        score_str = "inf" if r["score"] == float("inf") else f"{r['score']:.2f}"
        mark = "*" if r["label"] in frontier else ""
        print(f"{r['label']:28s} {r['accuracy']*100:8.1f}% {r['correct']:4d}/{r['tasks']:<4d} "
              f"{r['total_cost_usd']:12.6f} {score_str:>14s}  {mark}")
    print()
    print(f"Pareto frontier ({len(frontier)} configs, marked *): "
          f"{', '.join(sorted(frontier))}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures-dir", default="fixtures",
                        help="Directory containing sample_tasks.jsonl + expected.jsonl "
                             "(default: ./fixtures/)")
    parser.add_argument("--fake-transport", action="store_true",
                        help="Force the built-in offline fake transport even if provider "
                             "credentials are set. Auto-enabled when none are found.")
    args = parser.parse_args()

    fixtures_dir = Path(args.fixtures_dir)
    tasks, expected = load_fixtures(fixtures_dir)
    print(f"Loaded {len(tasks)} tasks from {fixtures_dir}")

    base = Settings.from_env()
    use_fake = args.fake_transport or not (base.groq_api_key or base.fireworks_api_key)
    if use_fake:
        print("No provider credentials found (or --fake-transport set): using the offline "
              "fake transport and forcing use_cloud_fallback=False to skip the network health "
              "check.")
        base = replace(base, use_cloud_fallback=False)
    transport = build_fake_transport(tasks, expected) if use_fake else None

    with tempfile.TemporaryDirectory(prefix="evaluate_routing_logs_") as tmp:
        log_dir = Path(tmp)

        configs: list[tuple[str, Settings]] = [("baseline (defaults)", base)]
        for param, values in SWEEP_RANGES.items():
            for value in values:
                configs.append((f"{param}={value}", replace(base, **{param: value})))

        results = []
        for label, settings in configs:
            result = run_config(tasks, expected, settings, transport, log_dir, label)
            results.append(result)
            score_str = "inf" if result["score"] == float("inf") else f"{result['score']:.2f}"
            print(f"  ran {label:28s} accuracy={result['accuracy']*100:5.1f}% "
                  f"cost=${result['total_cost_usd']:.6f} score={score_str}")

    print()
    print_table(results)


if __name__ == "__main__":
    main()
