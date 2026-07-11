"""Stage 1 logic tool: z3, solver-shaped problems only. Verifier = SAT/proof returned."""

from router.tools.logic_tool import try_z3

UNSAT_PROMPT = """Is the following set of constraints satisfiable?
```smtlib
(declare-const x Int)
(assert (> x 5))
(assert (< x 3))
```"""

SAT_PROMPT = """Determine satisfiability:
```smtlib
(declare-const x Int)
(assert (> x 5))
(assert (< x 10))
```"""


def test_unsat_constraints():
    r = try_z3(UNSAT_PROMPT)
    assert r.verified
    assert r.answer == "unsat"


def test_sat_constraints():
    r = try_z3(SAT_PROMPT)
    assert r.verified
    assert r.answer == "sat"


def test_natural_language_logic_is_not_solver_shaped():
    r = try_z3("If all cats are mammals and Tom is a cat, is Tom a mammal?")
    assert not r.verified
    assert r.answer is None


def test_malformed_smtlib_escalates():
    r = try_z3("```smtlib\n(assert (> x\n```")
    assert not r.verified
