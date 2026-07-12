"""CLI entrypoint for the batch routing job.

Pure wiring: parse args, load Settings, build the Phase 1 pipeline via the
factory, run the batch, print the summary. No routing decisions live here —
those all belong to the pipeline stages.

The grading harness runs the container with ZERO arguments, mounting the
dataset at /input and collecting /output, so both paths default to the
harness contract and stay overridable for local runs:

    python -m router.main                                       # harness mode
    python -m router.main --input tasks.jsonl --output out.jsonl  # local mode
"""

from __future__ import annotations

import argparse
from collections.abc import Callable

from router.batch import run_batch
from router.config import Settings
from router.factory import build_pipeline
from router.pipeline import Pipeline


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="router",
        description="Run the token-efficient LLM routing agent over a batch dataset.",
    )
    parser.add_argument("--input", default="/input/tasks.json",
                        help="Path to the input dataset (JSONL or CSV). "
                             "Defaults to the grading-harness mount.")
    parser.add_argument("--output", default="/output/results.json",
                        help="Path to write the results JSONL. "
                             "Defaults to the grading-harness mount.")
    return parser.parse_args(argv)


def _format_summary(summary: dict) -> str:
    return (
        "Batch complete:\n"
        f"  tasks:          {summary['tasks']}\n"
        f"  errors:         {summary['errors']}\n"
        f"  total_cost_usd: ${summary['total_cost_usd']:.6f}"
    )


def main(
    argv: list[str] | None = None,
    *,
    load_settings: Callable[[], Settings] = Settings.from_env,
    pipeline_builder: Callable[[Settings], Pipeline] = build_pipeline,
) -> dict:
    args = _parse_args(argv)
    settings = load_settings()
    pipeline = pipeline_builder(settings)
    summary = run_batch(pipeline, args.input, args.output)
    print(_format_summary(summary))
    return summary


if __name__ == "__main__":
    main()
