"""Stage 1 — deterministic tools, free, no LLM.

Returns an unverified StageResult for domains with no deterministic tool
(factual knowledge, summarization, code generation) so the cascade escalates.
"""

from __future__ import annotations

from router.tools.logic_tool import try_z3
from router.tools.math_tool import solve_math
from router.tools.ner_tool import NerTool
from router.tools.sandbox import run_python
from router.tools.sentiment_tool import classify_sentiment, extract_target_text
from router.types import Domain, StageResult, Task, ToolResult

STAGE = "stage1_tool"


class Stage1Deterministic:
    def __init__(self, ner_tool: NerTool | None = None, sentiment_threshold: float = 0.5,
                 sandbox_timeout: float = 2.0):
        self.ner_tool = ner_tool if ner_tool is not None else NerTool.load()
        self.sentiment_threshold = sentiment_threshold
        self.sandbox_timeout = sandbox_timeout

    def attempt(self, task: Task, domain: Domain) -> StageResult:
        handler = {
            Domain.MATH: self._math,
            Domain.SENTIMENT: self._sentiment,
            Domain.NER: self._ner,
            Domain.LOGIC: self._logic,
            Domain.CODE_DEBUG: self._code_debug,
        }.get(domain)
        if handler is None:
            return StageResult(None, STAGE, verified=False,
                               verifier_reason=f"no deterministic tool for {domain.value}")
        return handler(task)

    @staticmethod
    def _from_tool(result: ToolResult, verifier: str) -> StageResult:
        return StageResult(result.answer, STAGE, verified=result.verified,
                           verifier=verifier, verifier_reason=result.reason)

    def _math(self, task: Task) -> StageResult:
        return self._from_tool(solve_math(task.prompt), "sympy re-execution")

    def _sentiment(self, task: Task) -> StageResult:
        text = extract_target_text(task.prompt)
        result = classify_sentiment(text, threshold=self.sentiment_threshold)
        return self._from_tool(result, "VADER confidence threshold")

    def _ner(self, task: Task) -> StageResult:
        return self._from_tool(self.ner_tool.extract(extract_target_text(task.prompt)),
                               "spaCy span/label schema")

    def _logic(self, task: Task) -> StageResult:
        return self._from_tool(try_z3(task.prompt), "z3 SAT/proof")

    def _code_debug(self, task: Task) -> StageResult:
        code, tests = task.metadata.get("code"), task.metadata.get("tests")
        if not code or not tests:
            return StageResult(None, STAGE, verified=False,
                               verifier_reason="no code/tests in task metadata")
        run = run_python(code, tests=tests, timeout=self.sandbox_timeout)
        if run.passed:
            return StageResult(code, STAGE, verified=True, verifier="sandboxed test execution",
                               verifier_reason="provided code already passes its tests")
        return StageResult(None, STAGE, verified=False, verifier="sandboxed test execution",
                           verifier_reason=f"tests fail on provided code: {run.stderr[-400:]}")
