#!/usr/bin/env python3
"""PostToolUse hook: run the protocol + UCCI-adapter tests when relevant files change.

Triggered after Edit/Write. Reads the hook payload from stdin, inspects the edited
file path, and — only if it falls under ``src/dongfeng/protocol`` or
``src/dongfeng/engines`` — runs the focused conformance/adapter test subset.

Resilient by design: never hard-fails the tool call. If the payload can't be
parsed, the path is irrelevant, ``uv`` is missing, or the tests are absent, it
exits 0 quietly.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys

WATCHED = (
    os.path.join("src", "dongfeng", "protocol"),
    os.path.join("src", "dongfeng", "engines"),
)

# Only run tests that actually exist yet; missing files are skipped, not errors.
CANDIDATE_TESTS = (
    os.path.join("tests", "test_protocol_conformance.py"),
    os.path.join("tests", "test_ucci_adapter.py"),
)


def _edited_path(payload: dict) -> str:
    tool_input = payload.get("tool_input") or {}
    return str(tool_input.get("file_path") or tool_input.get("path") or "")


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except Exception:
        return 0  # no/garbled payload: nothing to do

    path = _edited_path(payload)
    if not path:
        return 0

    norm = os.path.normpath(path)
    if not any(w in norm for w in WATCHED):
        return 0  # not a protocol/engines file: skip

    project_dir = os.environ.get("CLAUDE_PROJECT_DIR", os.getcwd())
    tests = [t for t in CANDIDATE_TESTS if os.path.exists(os.path.join(project_dir, t))]
    if not tests:
        return 0  # tests not written yet (early milestones)

    if not shutil.which("uv"):
        return 0  # toolchain not available: don't block editing

    try:
        subprocess.run(
            ["uv", "run", "pytest", "-q", *tests],
            cwd=project_dir,
            check=False,
            timeout=300,
        )
    except Exception:
        return 0  # never hard-fail the edit on a test-runner problem

    return 0


if __name__ == "__main__":
    sys.exit(main())
