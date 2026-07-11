"""Stage 1 math tool: sympy-backed, verifier = re-execution / substitution check."""

from router.tools.math_tool import solve_math


def test_plain_arithmetic():
    r = solve_math("What is 12 * (3 + 4)?")
    assert r.verified
    assert r.answer == "84"


def test_arithmetic_with_caret_power():
    r = solve_math("Compute 2^10.")
    assert r.verified
    assert r.answer == "1024"


def test_linear_equation_with_implicit_multiplication():
    r = solve_math("Solve for x: 3x + 7 = 22")
    assert r.verified
    assert r.answer == "5"


def test_no_math_content_escalates():
    r = solve_math("Tell me about the French Revolution.")
    assert not r.verified
    assert r.answer is None


def test_does_not_execute_arbitrary_code():
    r = solve_math("What is __import__('os').system('echo pwned')?")
    assert not r.verified
