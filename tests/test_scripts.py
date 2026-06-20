import os
import subprocess
import sys

import pytest

SCRIPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "notebooks")


def get_all_scripts():
    return sorted(
        os.path.join(SCRIPT_DIR, f) for f in os.listdir(SCRIPT_DIR) if f.endswith(".py")
    )


def _is_parked(script_path):
    """A parked script carries the documented ``# --- PARKED`` marker.

    Parked scripts deliberately ``raise NotImplementedError`` (e.g. because
    they depend on data we don't have), so they cannot run top-to-bottom. The
    marker is the single source of truth: when a script is un-parked (marker
    and guard removed) it is automatically held to the normal must-run bar
    again, with no change needed here.
    """
    with open(script_path, encoding="utf-8") as f:
        return "# --- PARKED" in f.read()


def _script_params():
    params = []
    for script_path in get_all_scripts():
        if _is_parked(script_path):
            params.append(
                pytest.param(
                    script_path,
                    marks=pytest.mark.xfail(
                        reason=(
                            "Parked script: deliberately raises "
                            "NotImplementedError until re-pointed at real data"
                        ),
                        strict=True,
                    ),
                )
            )
        else:
            params.append(script_path)
    return params


@pytest.mark.parametrize("script_path", _script_params())
def test_script_execution(script_path):
    """Every jupytext script in ``notebooks/`` must run top-to-bottom.

    Scripts are run from ``notebooks/`` (so the ``sys.path.insert(0, "..")``
    they use points at the repo root) with a headless matplotlib backend.

    A script parked behind a ``# --- PARKED`` marker / ``raise
    NotImplementedError`` guard (e.g. because it depends on data we don't
    have) is expected to fail and is marked ``xfail`` (see ``_script_params``).
    The xfail is ``strict``, so if a parked script ever runs to completion the
    test fails -- a prompt to remove the stale PARKED marker.
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
