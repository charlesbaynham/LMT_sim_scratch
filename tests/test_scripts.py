import os
import subprocess
import sys

import pytest

SCRIPT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "notebooks"
)


def get_all_scripts():
    return sorted(
        os.path.join(SCRIPT_DIR, f)
        for f in os.listdir(SCRIPT_DIR)
        if f.endswith(".py")
    )


@pytest.mark.parametrize("script_path", get_all_scripts())
def test_script_execution(script_path):
    """Every jupytext script in ``notebooks/`` must run top-to-bottom.

    Scripts are run from ``notebooks/`` (so the ``sys.path.insert(0, "..")``
    they use points at the repo root) with a headless matplotlib backend.

    A script that cannot currently run -- e.g. one parked behind a
    ``raise NotImplementedError`` guard because it depends on data we don't
    have -- is NOT exempted: it fails this test on purpose. A parked script is
    a known-broken state, and a red test is exactly how it should surface until
    it is fixed or removed.
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

    assert result.returncode == 0, (
        f"Script {script_path} failed to execute:\n{result.stderr[-3000:]}"
    )
