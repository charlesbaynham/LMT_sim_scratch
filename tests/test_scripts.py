import os
import subprocess
import sys

import pytest

SCRIPT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "notebooks"
)

# Marker emitted by the guard block of a deliberately parked script (one fenced
# off behind ``raise NotImplementedError`` because it runs on legacy/placeholder
# data). Such scripts are EXPECTED to raise rather than execute cleanly.
PARKED_MARKER = "PARKED:"


def get_all_scripts():
    return sorted(
        os.path.join(SCRIPT_DIR, f)
        for f in os.listdir(SCRIPT_DIR)
        if f.endswith(".py")
    )


def _is_parked(path):
    with open(path, "r", encoding="utf-8") as f:
        return PARKED_MARKER in f.read()


@pytest.mark.parametrize("script_path", get_all_scripts())
def test_script_execution(script_path):
    """Every jupytext script in ``notebooks/`` must run top-to-bottom.

    Scripts are run from ``notebooks/`` (so the ``sys.path.insert(0, "..")``
    they use points at the repo root) with a headless matplotlib backend.
    Scripts deliberately parked behind a ``raise NotImplementedError`` guard are
    the exception: they must raise that guard (and nothing else) instead of
    finishing cleanly.
    """
    env = {**os.environ, "MPLBACKEND": "Agg"}
    result = subprocess.run(
        [sys.executable, os.path.basename(script_path)],
        cwd=SCRIPT_DIR,
        capture_output=True,
        text=True,
        timeout=600,
        env=env,
    )

    if _is_parked(script_path):
        assert result.returncode != 0, (
            f"Parked script {script_path} exited cleanly but should raise its "
            f"NotImplementedError guard."
        )
        assert "NotImplementedError" in result.stderr, (
            f"Parked script {script_path} raised an unexpected error instead of "
            f"its NotImplementedError guard:\n{result.stderr[-3000:]}"
        )
        return

    assert result.returncode == 0, (
        f"Script {script_path} failed to execute:\n{result.stderr[-3000:]}"
    )
