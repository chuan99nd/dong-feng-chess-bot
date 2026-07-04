"""Tests for the board-state dataset pipeline (WP1 / M3.5)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from dongfeng.core import STARTING_FEN, Color, GameResult, Move, new_board
from dongfeng.data import BoardBuildStats, Game, build_board_shards, load_board_arrays
from dongfeng.tokenizer.board_tokenizer import BoardTokenizer

# ---------------------------------------------------------------------------
# Shared test fixtures
# ---------------------------------------------------------------------------

# Four moves: two Red, two Black — a minimal playable sequence.
_MOVES = [Move.from_iccs(s) for s in ("h2e2", "h9g7", "h0g2", "i9h9")]


def _game(result: GameResult = GameResult.RED_WIN) -> Game:
    return Game(start_fen=STARTING_FEN, moves=list(_MOVES), result=result)


# ---------------------------------------------------------------------------
# Test: counts match total plies
# ---------------------------------------------------------------------------


def test_build_board_shards_counts(tmp_path: Path) -> None:
    """num_samples must equal total plies across all games."""
    stats = build_board_shards([_game(), _game()], tmp_path / "ds")
    assert isinstance(stats, BoardBuildStats)
    assert stats.num_games == 2
    assert stats.num_samples == 2 * len(_MOVES)
    assert stats.skipped_games == 0
    assert len(stats.shards) == 1  # fits in one shard


# ---------------------------------------------------------------------------
# Test: board decode round-trips to the correct FEN
# ---------------------------------------------------------------------------


def test_board_decode_matches_replayed_fen(tmp_path: Path) -> None:
    """BoardTokenizer.decode of a sample's board row must match the FEN of the
    replayed board at that ply (before the move was pushed)."""
    out = tmp_path / "ds"
    build_board_shards([_game()], out)
    boards, _moves, _values = load_board_arrays(out)

    board_tok = BoardTokenizer()

    # Replay the game and collect FENs before each move.
    expected_fens: list[str] = []
    board = new_board(STARTING_FEN)
    for move in _MOVES:
        expected_fens.append(board.fen())
        board.push(move)

    assert len(expected_fens) == len(_MOVES)
    assert boards.shape[0] == len(_MOVES)

    # Check every sample — the decoded board must match the replayed FEN
    # (short two-field form: placement + side).
    # BoardTokenizer.decode returns 'r'/'b' for the side; the FEN may use 'w'
    # as an alias for Red ('r'), so normalise before comparing.
    def _normalise_side(s: str) -> str:
        parts = s.split()
        if len(parts) == 2 and parts[1] == "w":
            return f"{parts[0]} r"
        return s

    for idx, fen in enumerate(expected_fens):
        decoded = board_tok.decode(boards[idx].tolist())
        fen_short = _normalise_side(" ".join(fen.split()[:2]))
        assert decoded == fen_short, f"Sample {idx}: decoded {decoded!r} != expected {fen_short!r}"


# ---------------------------------------------------------------------------
# Test: value signs for RED_WIN
# ---------------------------------------------------------------------------


def test_value_signs_red_win(tmp_path: Path) -> None:
    """In a RED_WIN game: Red-to-move plies → +1, Black-to-move plies → −1."""
    out = tmp_path / "ds_redwin"
    build_board_shards([_game(GameResult.RED_WIN)], out)
    boards, _moves, values = load_board_arrays(out)

    # Determine the expected turn for each ply by replaying.
    turns: list[Color] = []
    board = new_board(STARTING_FEN)
    for move in _MOVES:
        turns.append(board.turn)
        board.push(move)

    for idx, turn in enumerate(turns):
        v = int(values[idx])
        if turn is Color.RED:
            assert v == 1, f"Ply {idx} (RED to move, RED_WIN): expected +1, got {v}"
        else:
            assert v == -1, f"Ply {idx} (BLACK to move, RED_WIN): expected -1, got {v}"


# ---------------------------------------------------------------------------
# Test: value sentinel for ONGOING
# ---------------------------------------------------------------------------


def test_value_sentinel_ongoing(tmp_path: Path) -> None:
    """ONGOING games must produce value=127 (mask sentinel) for all plies."""
    out = tmp_path / "ds_ongoing"
    build_board_shards([_game(GameResult.ONGOING)], out)
    _boards, _moves, values = load_board_arrays(out)
    assert all(int(v) == 127 for v in values), "Expected all values == 127 for ONGOING"


# ---------------------------------------------------------------------------
# Test: board_meta.json schema matches §1.3
# ---------------------------------------------------------------------------


def test_board_meta_json_schema(tmp_path: Path) -> None:
    """board_meta.json must contain all keys specified in §1.3."""
    out = tmp_path / "ds_meta"
    build_board_shards([_game()], out, created="2026-07-03T00:00:00+00:00")
    meta = json.loads((out / "board_meta.json").read_text())

    required_keys = {
        "schema",
        "num_samples",
        "num_games",
        "skipped_games",
        "tokenizer",
        "move_tokenizer",
        "shards",
        "created",
    }
    assert required_keys <= meta.keys(), f"Missing keys: {required_keys - meta.keys()}"
    assert meta["schema"] == "board-ds-v1"
    assert meta["tokenizer"] == "board-v1"
    assert meta["move_tokenizer"] == "move-v1"
    assert meta["created"] == "2026-07-03T00:00:00+00:00"
    assert meta["num_samples"] == len(_MOVES)
    assert meta["num_games"] == 1
    assert meta["skipped_games"] == 0
    assert isinstance(meta["shards"], list)


# ---------------------------------------------------------------------------
# Test: load_board_arrays returns correct dtypes and shapes
# ---------------------------------------------------------------------------


def test_load_board_arrays_shapes_and_dtypes(tmp_path: Path) -> None:
    """load_board_arrays must return arrays with the right dtypes and shapes."""
    out = tmp_path / "ds_shapes"
    stats = build_board_shards([_game(), _game()], out)
    boards, moves, values = load_board_arrays(out)

    n = stats.num_samples
    assert boards.shape == (n, 91), f"boards shape {boards.shape} != ({n}, 91)"
    assert moves.shape == (n,), f"moves shape {moves.shape} != ({n},)"
    assert values.shape == (n,), f"values shape {values.shape} != ({n},)"

    assert boards.dtype == np.uint8, f"boards dtype {boards.dtype} != uint8"
    assert moves.dtype == np.uint16, f"moves dtype {moves.dtype} != uint16"
    assert values.dtype == np.int8, f"values dtype {values.dtype} != int8"


# ---------------------------------------------------------------------------
# Test: shard rolling
# ---------------------------------------------------------------------------


def test_shard_rolling(tmp_path: Path) -> None:
    """Shard rolling: buffer flushes after each game when shard_size is exceeded.

    With shard_size=2 and 4-ply games: after game 1 the buffer has 4 >= 2 and
    flushes (1 shard). After game 2 same (1 more shard). Total: 2 shards.
    """
    out = tmp_path / "ds_roll"
    stats = build_board_shards([_game(), _game()], out, shard_size=2)
    assert stats.num_samples == 2 * len(_MOVES)  # 8 total
    # Each game fills > shard_size and flushes: 2 shards
    assert len(stats.shards) == 2

    boards, moves, values = load_board_arrays(out)
    assert boards.shape[0] == stats.num_samples


# ---------------------------------------------------------------------------
# Test: skipped games
# ---------------------------------------------------------------------------


def test_skipped_illegal_game(tmp_path: Path) -> None:
    """A game with an illegal first move should be counted as skipped."""
    bad_game = Game(
        start_fen=STARTING_FEN,
        moves=[Move.from_iccs("a0a0")],  # a0->a0 is never legal
        result=GameResult.ONGOING,
    )
    out = tmp_path / "ds_skip"
    stats = build_board_shards([bad_game, _game()], out)
    assert stats.skipped_games == 1
    assert stats.num_games == 1
    assert stats.num_samples == len(_MOVES)


# ---------------------------------------------------------------------------
# Test: draw value
# ---------------------------------------------------------------------------


def test_value_draw(tmp_path: Path) -> None:
    """DRAW games must produce value=0 for all plies."""
    out = tmp_path / "ds_draw"
    build_board_shards([_game(GameResult.DRAW)], out)
    _boards, _moves, values = load_board_arrays(out)
    assert all(int(v) == 0 for v in values), "Expected all values == 0 for DRAW"
