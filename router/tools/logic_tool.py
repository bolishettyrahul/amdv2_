"""Stage 1 logic tool — z3, applied only to solver-shaped problems.

Natural-language logic puzzles are NOT parsed here; anything without an
explicit SMT-LIB constraint block escalates to Stage 2/3.
"""

from __future__ import annotations

import re

from router.types import ToolResult

_SMTLIB_BLOCK = re.compile(r"```(?:smtlib|smt2?|lisp)?\s*\n(.*?)```", re.S)


def _find_smtlib(prompt: str) -> str | None:
    for block in _SMTLIB_BLOCK.findall(prompt):
        if "(assert" in block:
            return block
    # Bare SMT-LIB without code fences.
    if "(assert" in prompt and "(declare-" in prompt:
        start = prompt.index("(declare-")
        return prompt[start:]
    return None


def try_z3(prompt: str) -> ToolResult:
    source = _find_smtlib(prompt)
    if source is None:
        return ToolResult(None, verified=False, reason="not solver-shaped (no SMT-LIB block)")
    import z3

    try:
        solver = z3.Solver()
        solver.add(z3.parse_smt2_string(source))
        outcome = solver.check()
    except z3.Z3Exception as exc:
        return ToolResult(None, verified=False, reason=f"z3 parse/solve error: {exc}")
    if outcome == z3.sat:
        return ToolResult("sat", verified=True, confidence=1.0,
                          reason=f"z3 model: {solver.model()}")
    if outcome == z3.unsat:
        return ToolResult("unsat", verified=True, confidence=1.0, reason="z3 proved unsat")
    return ToolResult(None, verified=False, reason="z3 returned unknown")
