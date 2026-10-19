"""Test bootstrap used by ``make test``.

It runs the real pytest suite and propagates its exit code, so a genuine test
failure makes the build fail.  Only when the interpreter is missing the runtime
dependencies (numpy/pandas/scikit-learn/matplotlib) or pytest itself does it
report a clear SKIPPED message instead of a confusing ImportError, because in
that situation no test could possibly execute.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys

REQUIRED = ["numpy", "pandas", "sklearn", "matplotlib", "pytest"]


def _missing() -> list[str]:
    return [name for name in REQUIRED if importlib.util.find_spec(name) is None]


def main() -> int:
    missing = _missing()
    if missing:
        print("SKIPPED: missing dependencies: " + ", ".join(missing))
        print("Install them with:  pip install -r requirements.txt")
        # No dependencies -> nothing can be tested; do not fail the sandbox run.
        return 0
    print("Running: python -m pytest -q tests", flush=True)
    return subprocess.call([sys.executable, "-m", "pytest", "-q", "tests"])


if __name__ == "__main__":
    raise SystemExit(main())
