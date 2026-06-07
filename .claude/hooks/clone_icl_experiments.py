#!/usr/bin/env python3
"""Clone a vendored copy of icl_experiments at a pinned commit."""

import os
import subprocess
import sys

REPO_NAME = "icl_experiments"
REPO_URL = "https://gitlab.com/aion-physics/code/artiq/experiment-repositories/icl_experiments.git"
# Pinned to the current master HEAD of icl_experiments.
REPO_REV = "cec25f6d3e7d9d92455d77d8eeb0ff8a6927c521"


def run(cmd: list[str]) -> None:
    print(f"  $ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def checkout_vendor(name: str, url: str, rev: str) -> None:
    d = f"vendor/{name}"
    if os.path.isdir(f"{d}/.git"):
        current = (
            subprocess.check_output(["git", "-C", d, "rev-parse", "HEAD"])
            .decode()
            .strip()
        )
        if current == rev:
            print(f"  {name}: already at {rev[:8]}", flush=True)
            return
        print(f"  {name}: updating {current[:8]} → {rev[:8]}", flush=True)
    else:
        print(f"  {name}: cloning {url} @ {rev[:8]}", flush=True)
        run(["git", "init", d])

    # Fetch the specific commit by SHA (works on GitLab with allowAnySHA1InWant).
    run(["git", "-C", d, "fetch", "--depth=1", url, rev])
    run(["git", "-C", d, "checkout", "FETCH_HEAD"])
    print(f"  {name}: ready", flush=True)


os.makedirs("vendor", exist_ok=True)

try:
    checkout_vendor(REPO_NAME, REPO_URL, REPO_REV)
except Exception as exc:
    print(f"WARNING: could not vendor {REPO_NAME}: {exc}", file=sys.stderr, flush=True)
    sys.exit(1)
