"""Subprocess sandbox for the code domains.

Executes untrusted generated code + tests in an isolated interpreter
(`python -I`) inside a throwaway temp directory with a hard timeout, so runs
cannot leak state into the batch job's working directory or hang the pipeline.
Memory limits are applied on POSIX (the grading sandbox); Windows dev machines
rely on the timeout alone.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

DEFAULT_TIMEOUT = 2.0
_MEMORY_LIMIT_BYTES = 512 * 1024 * 1024


@dataclass(frozen=True)
class SandboxResult:
    passed: bool
    stdout: str
    stderr: str
    returncode: int | None
    timed_out: bool


def _posix_limits():
    import resource

    resource.setrlimit(resource.RLIMIT_AS, (_MEMORY_LIMIT_BYTES, _MEMORY_LIMIT_BYTES))


def _decode(data) -> str:
    if data is None:
        return ""
    return data.decode(errors="replace") if isinstance(data, bytes) else data


def run_python(code: str, tests: str | None, timeout: float = DEFAULT_TIMEOUT) -> SandboxResult:
    source = code if tests is None else f"{code}\n\n{tests}\n"
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        script = Path(tmpdir) / "main.py"
        script.write_text(source, encoding="utf-8")
        kwargs: dict = {}
        if os.name == "posix":
            kwargs["preexec_fn"] = _posix_limits
        try:
            proc = subprocess.run(
                [sys.executable, "-I", str(script)],
                cwd=tmpdir,
                capture_output=True,
                text=True,
                timeout=timeout,
                **kwargs,
            )
        except subprocess.TimeoutExpired as exc:
            return SandboxResult(False, _decode(exc.stdout), _decode(exc.stderr), None, True)
    return SandboxResult(proc.returncode == 0, proc.stdout, proc.stderr, proc.returncode, False)
