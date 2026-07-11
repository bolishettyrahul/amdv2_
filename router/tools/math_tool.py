"""Stage 1 math tool — sympy-backed, no LLM.

Extracts an equation or arithmetic expression from the prompt, solves it with
sympy, and verifies by substitution / re-evaluation. Anything it cannot parse
confidently is left unverified so the cascade escalates.
"""

from __future__ import annotations

import re

from router.types import ToolResult

# Charset guard: expressions may only contain digits, simple variable names,
# whitespace and arithmetic operators — nothing that could smuggle code into
# the parser (quotes, underscores, attribute access).
_SAFE = re.compile(r"^[0-9a-zA-Z\s.+\-*/^()=]+$")
_EQUATION = re.compile(r"([0-9a-zA-Z\s.+\-*/^()]+=[0-9a-zA-Z\s.+\-*/^()]+)")
_ARITHMETIC = re.compile(r"[\d(][\d\s.+\-*/^()]*[\d)]")


def _parse(text: str):
    from sympy.parsing.sympy_parser import (
        convert_xor,
        implicit_multiplication_application,
        parse_expr,
        standard_transformations,
    )

    transforms = standard_transformations + (implicit_multiplication_application, convert_xor)
    return parse_expr(text, transformations=transforms, evaluate=True)


def _format_number(value) -> str:
    import sympy

    value = sympy.nsimplify(value, rational=False) if value.is_number else value
    if value.is_integer or (value.is_number and float(value) == int(float(value))):
        return str(int(float(value)))
    return str(value)


def _solve_equation(segment: str) -> ToolResult | None:
    import sympy

    lhs_s, _, rhs_s = segment.partition("=")
    try:
        lhs, rhs = _parse(lhs_s), _parse(rhs_s)
    except Exception:
        return None
    equation = sympy.Eq(lhs, rhs)
    free = sorted(equation.free_symbols, key=str)
    if len(free) != 1:
        return None
    try:
        solutions = sympy.solve(equation, free[0])
    except Exception:
        return None
    if len(solutions) != 1:
        return None
    sol = solutions[0]
    # Verifier: substitute the solution back into the original equation.
    if sympy.simplify(equation.subs(free[0], sol)) is not sympy.true:
        return ToolResult(None, verified=False, reason="substitution check failed")
    return ToolResult(_format_number(sol), verified=True, confidence=1.0,
                      reason=f"sympy solved {segment.strip()} and substitution holds")


def _eval_arithmetic(candidate: str) -> ToolResult | None:
    try:
        expr = _parse(candidate)
    except Exception:
        return None
    if not expr.is_number:
        return None
    # Verifier: independent numeric re-evaluation must agree with exact result.
    if abs(float(expr.evalf()) - float(expr)) > 1e-9:
        return ToolResult(None, verified=False, reason="re-evaluation mismatch")
    return ToolResult(_format_number(expr), verified=True, confidence=1.0,
                      reason=f"sympy evaluated {candidate.strip()}")


def solve_math(prompt: str) -> ToolResult:
    for match in _EQUATION.finditer(prompt):
        segment = match.group(1)
        if not _SAFE.match(segment) or not any(c.isdigit() for c in segment):
            continue
        result = _solve_equation(segment)
        if result is not None:
            return result

    candidates = [m.group(0) for m in _ARITHMETIC.finditer(prompt)
                  if _SAFE.match(m.group(0)) and re.search(r"[+\-*/^]", m.group(0))]
    for candidate in sorted(candidates, key=len, reverse=True):
        result = _eval_arithmetic(candidate)
        if result is not None:
            return result

    return ToolResult(None, verified=False, reason="no parseable math expression found")
