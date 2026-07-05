"""Tests for Pikafish score parsing + the resumable label-eval pipeline."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from dongfeng.data import build_board_shards
from dongfeng.data.label_eval import eval_to_value, label_eval
from dongfeng.engines.pikafish_engine import PikafishEngine, PikafishEval

# Reuse the tiny-game fixture builder from the training tests.
from test_board_training import _make_games  # type: ignore[import-not-found]


@pytest.mark.parametrize(
    "line,expected",
    [
        ("info depth 18 seldepth 24 multipv 1 score cp 34 nodes 100 pv h2e2", (34, None)),
        ("info depth 20 score mate 3 pv b0c2", (None, 3)),
        ("info depth 20 score cp -128 pv x", (-128, None)),
        ("info depth 1 score cp 0 pv", (0, None)),
        ("info string loading", (None, None)),
        ("bestmove h2e2 ponder h9g7", (None, None)),
    ],
)
def test_parse_info_score(line: str, expected: tuple[int | None, int | None]) -> None:
    assert PikafishEngine._parse_info_score(line) == expected


def test_eval_to_value() -> None:
    assert eval_to_value(PikafishEval(mate=2), 300.0) == 1.0
    assert eval_to_value(PikafishEval(mate=-1), 300.0) == -1.0
    assert eval_to_value(PikafishEval(cp=300), 300.0) == pytest.approx(math.tanh(1.0))
    assert eval_to_value(PikafishEval(cp=0), 300.0) == 0.0
    assert math.isnan(eval_to_value(PikafishEval(), 300.0))


def test_label_eval_writes_and_resumes(tmp_path: Path) -> None:
    ds = tmp_path / "ds"
    build_board_shards(_make_games(6), ds)
    meta = json.loads((ds / "board_meta.json").read_text())
    total = meta["num_samples"]
    stem = meta["shards"][0]

    calls = {"n": 0}

    def fake(fen: str, depth: int) -> PikafishEval:
        calls["n"] += 1
        return PikafishEval(cp=20)

    sd = tmp_path / "label-x"
    res = label_eval(ds, evaluator=fake, depth=12, status_dir=sd, flush_every=4)
    assert res["done"] == total
    assert res["status"] == "done"

    arr = np.fromfile(ds / f"values_eval_{stem}.bin", dtype=np.float32)
    assert arr.size == total
    assert not np.isnan(arr).any()
    assert arr.min() == pytest.approx(math.tanh(20 / 300.0), abs=1e-6)

    # Re-run resumes at the end → no new evaluator calls.
    before = calls["n"]
    label_eval(ds, evaluator=fake, depth=12, status_dir=sd, flush_every=4)
    assert calls["n"] == before
