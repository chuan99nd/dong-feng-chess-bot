"""Tests for sample explosion and shard building (M1)."""

from __future__ import annotations

import json
from array import array
from pathlib import Path

from dongfeng.core import STARTING_FEN, Color, GameResult, Move
from dongfeng.data import Game, build_shards, iter_samples

_MOVES = [Move.from_iccs(s) for s in ("h2e2", "h9g7", "h0g2", "i9h9")]


def _game(result: GameResult = GameResult.RED_WIN) -> Game:
    return Game(start_fen=STARTING_FEN, moves=list(_MOVES), result=result)


def test_iter_samples_replays_all_plies() -> None:
    samples = list(iter_samples([_game()]))
    assert len(samples) == len(_MOVES)
    # First sample is the start position, Red to move.
    assert samples[0].fen.split()[0] == STARTING_FEN.split()[0]
    assert samples[0].turn is Color.RED
    assert samples[0].move == _MOVES[0]


def test_winning_side_only_filter() -> None:
    samples = list(iter_samples([_game(GameResult.RED_WIN)], winning_side_only=True))
    # Only Red's moves (plies 1 and 3 of 4) survive.
    assert len(samples) == 2
    assert all(s.turn is Color.RED for s in samples)


def test_build_shards_writes_bin_and_meta(tmp_path: Path) -> None:
    out = tmp_path / "ds"
    stats = build_shards([_game(), _game()], out, created="2026-07-01T00:00:00+00:00")

    assert stats.num_games == 2
    assert stats.num_samples == 2 * len(_MOVES)
    # Each game = BOS + N moves + EOS.
    assert stats.num_tokens == 2 * (len(_MOVES) + 2)
    assert stats.num_shards == 1
    assert stats.skipped_games == 0

    meta = json.loads((out / "dataset_meta.json").read_text())
    assert meta["num_games"] == 2
    assert meta["tokenizer"] == "move-v1"
    assert meta["created"] == "2026-07-01T00:00:00+00:00"

    shard = Path(stats.shard_paths[0])
    assert shard.exists()
    data = array("H")
    data.frombytes(shard.read_bytes())
    assert len(data) == stats.num_tokens


def test_build_shards_skips_empty_games(tmp_path: Path) -> None:
    empty = Game(start_fen=STARTING_FEN, moves=[], result=GameResult.DRAW)
    stats = build_shards([empty], tmp_path / "ds2")
    assert stats.num_games == 0
    assert stats.skipped_games == 1
    assert stats.num_shards == 0
