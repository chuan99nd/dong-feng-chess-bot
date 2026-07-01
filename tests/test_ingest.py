"""Tests for game-record ingestion (M1)."""

from __future__ import annotations

from pathlib import Path

import pytest

from dongfeng.core import GameResult
from dongfeng.data import parse_file, parse_pgn

_SAMPLE_PGN = """[Game "China Chess"]
[Event "Test Event"]
[Red "Red Player"]
[Black "Black Player"]
[Result "1-0"]
1. 炮二平五 马8进7
2. 马二进三 车9平8
3. 车一平二 炮8进4
"""

_EXPECTED_ICCS = ["h2e2", "h9g7", "h0g2", "i9h9", "i0h0", "h7h3"]


def _write_pgn(tmp_path: Path) -> Path:
    p = tmp_path / "sample.pgn"
    p.write_text(_SAMPLE_PGN, encoding="utf-8")
    return p


def test_parse_pgn_moves_and_result(tmp_path: Path) -> None:
    games = list(parse_pgn(_write_pgn(tmp_path)))
    assert len(games) == 1
    game = games[0]
    assert [m.iccs for m in game.moves] == _EXPECTED_ICCS
    assert game.result is GameResult.RED_WIN
    assert game.metadata.get("event") == "Test Event"


def test_parse_file_dispatch(tmp_path: Path) -> None:
    games = list(parse_file(_write_pgn(tmp_path)))
    assert [m.iccs for m in games[0].moves] == _EXPECTED_ICCS


def test_unsupported_extension_raises(tmp_path: Path) -> None:
    bad = tmp_path / "game.unknown"
    bad.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError):
        list(parse_file(bad))
