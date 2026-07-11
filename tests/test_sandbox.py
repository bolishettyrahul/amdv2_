"""Code-domain sandbox: subprocess isolation, timeout, no state leakage."""

from router.tools.sandbox import run_python


def test_passing_code_and_tests():
    r = run_python("def add(a, b):\n    return a + b\n", tests="assert add(2, 3) == 5")
    assert r.passed
    assert not r.timed_out


def test_failing_assertion_reports_failure():
    r = run_python("def add(a, b):\n    return a - b\n", tests="assert add(2, 3) == 5")
    assert not r.passed
    assert "AssertionError" in r.stderr


def test_syntax_error_reports_failure():
    r = run_python("def broken(:\n    pass", tests=None)
    assert not r.passed
    assert "SyntaxError" in r.stderr


def test_infinite_loop_times_out():
    r = run_python("while True:\n    pass", tests=None, timeout=1.0)
    assert not r.passed
    assert r.timed_out


def test_runs_in_isolated_cwd_no_state_leakage(tmp_path):
    # Files written by the tested code must not land in the batch job's cwd.
    r1 = run_python("open('leak.txt', 'w').write('x')", tests=None)
    assert r1.passed
    r2 = run_python(
        "import os, sys\nsys.exit(1 if os.path.exists('leak.txt') else 0)", tests=None
    )
    assert r2.passed


def test_stdout_captured():
    r = run_python("print('hello world')", tests=None)
    assert r.passed
    assert "hello world" in r.stdout
