"""Smoke tests: every documented example must still run to completion.

They check no number, only that each script exits successfully -- the level
of verification the examples got manually during review, now automated.

Scripts run as subprocesses, in their own directory (they resolve data files
relative to their own location) and with a headless matplotlib backend, so
that a crash in one cannot affect the rest of the suite.

Markers:
    ``slow``     -- takes more than a few seconds; deselect with -m "not slow"
    ``veryslow`` -- takes many minutes; deselected by default (see pyproject)
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS = REPO_ROOT / "docs"

# Not an example in its own right: it exists to be imported for its side
# effect, putting the repository on sys.path.
_NOT_EXAMPLES = {"daceypy_import_helper.py"}

# Wall-clock class of the slower examples, measured on a developer machine.
# Anything not listed here runs in a few seconds at most.
_SLOW = {
    "docs/ADS/2Advanced-Ex.py",
    "docs/ADS/3OnlineADS-Ex.py",
    "docs/ADS/4OptimizedADS-Ex.py",
}
_VERY_SLOW = {
    # DA.init(15, 6) plus a 81x81 evaluation grid per frame: tens of minutes.
    "docs/Tutorials/Tutorial2/6Integrator-Ex.py",
}


def _examples() -> list[str]:
    found = sorted(
        p.relative_to(REPO_ROOT).as_posix()
        for p in DOCS.rglob("*.py")
        if p.name not in _NOT_EXAMPLES
    )
    assert found, "no example scripts discovered"
    return found


def _marks(rel: str):
    if rel in _VERY_SLOW:
        return [pytest.mark.slow, pytest.mark.veryslow]
    if rel in _SLOW:
        return [pytest.mark.slow]
    return []


@pytest.mark.parametrize(
    "rel", [pytest.param(rel, marks=_marks(rel)) for rel in _examples()]
)
def test_example_runs(rel: str):
    script = REPO_ROOT / rel
    env = {**os.environ, "MPLBACKEND": "Agg", "PYTHONPATH": str(REPO_ROOT)}
    result = subprocess.run(
        [sys.executable, script.name],
        cwd=script.parent,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"{rel} exited with {result.returncode}\n"
        f"--- stdout (tail) ---\n{result.stdout[-2000:]}\n"
        f"--- stderr (tail) ---\n{result.stderr[-2000:]}"
    )


@pytest.mark.parametrize("rel", _examples())
def test_example_text_is_ascii(rel: str):
    """No example may print characters the console cannot encode.

    On Windows ``sys.stdout`` defaults to the legacy code page, so a single
    arrow or Greek letter in a printed string aborts the script with
    ``UnicodeEncodeError`` -- for the user running the example, not just for
    this suite. Checking the string literals catches that on every platform,
    including for the examples that are too slow to run in the CI matrix.

    Comments and identifiers are exempt: they are decoded from the source as
    UTF-8 and never reach the console.
    """
    source = (REPO_ROOT / rel).read_text(encoding="utf-8")
    offenders = sorted(
        {
            char
            for node in ast.walk(ast.parse(source, filename=rel))
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
            for char in node.value
            if ord(char) > 127
        }
    )
    assert not offenders, (
        f"{rel} has non-ASCII characters in string literals: "
        + ", ".join(f"{c!r} (U+{ord(c):04X})" for c in offenders)
    )
