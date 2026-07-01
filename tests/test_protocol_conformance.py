"""RandomEngine must satisfy the universal engine conformance suite."""

from __future__ import annotations

from dongfeng.engines import RandomEngine
from dongfeng.protocol import run_conformance


def test_random_engine_conforms() -> None:
    """``run_conformance(RandomEngine)`` returns no failure messages."""
    failures = run_conformance(RandomEngine)
    assert failures == [], "conformance failures:\n" + "\n".join(failures)
