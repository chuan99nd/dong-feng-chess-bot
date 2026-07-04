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


# Augmentation is on by default; the non-augmentation tests below opt out so
# they can assert raw per-ply encoding/counts. Keyword bundle for that.
_NO_AUG = {"mirror": False, "color_augment": False}


# ---------------------------------------------------------------------------
# Test: counts match total plies
# ---------------------------------------------------------------------------


def test_build_board_shards_counts(tmp_path: Path) -> None:
    """num_samples must equal total plies across all games."""
    stats = build_board_shards([_game(), _game()], tmp_path / "ds", **_NO_AUG)
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
    build_board_shards([_game()], out, **_NO_AUG)
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
    build_board_shards([_game()], out, created="2026-07-03T00:00:00+00:00", **_NO_AUG)
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
    stats = build_board_shards([_game(), _game()], out, shard_size=2, **_NO_AUG)
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
    stats = build_board_shards([bad_game, _game()], out, **_NO_AUG)
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


# ===========================================================================
# WP-AUG: data augmentation (mirror + colour swap)
# ===========================================================================

from dongfeng.data.board_dataset import (  # noqa: E402
    COLOR_SWAP,
    MOVE_MIRROR,
    MOVE_ROT180,
    POS_MIRROR,
    POS_ROT180,
)
from dongfeng.tokenizer.move_tokenizer import MoveTokenizer  # noqa: E402


def test_pos_permutations_are_involutions() -> None:
    """Applying each board permutation twice is the identity, and index 90 fixed."""
    ident = np.arange(91)
    assert np.array_equal(POS_MIRROR[POS_MIRROR], ident)
    assert np.array_equal(POS_ROT180[POS_ROT180], ident)
    assert POS_MIRROR[90] == 90
    assert POS_ROT180[90] == 90


def test_color_swap_lut() -> None:
    """COLOR_SWAP is an involution mapping red<->black pieces and side tokens."""
    ident = np.arange(21)
    assert np.array_equal(COLOR_SWAP[COLOR_SWAP], ident)
    # Red pieces 5..11 <-> black 12..18 (black = red + 7).
    for red in range(5, 12):
        assert COLOR_SWAP[red] == red + 7
        assert COLOR_SWAP[red + 7] == red
    # Side tokens flip.
    assert COLOR_SWAP[19] == 20
    assert COLOR_SWAP[20] == 19
    # Specials + empty are unchanged.
    for t in (0, 1, 2, 3, 4):
        assert COLOR_SWAP[t] == t


def test_move_perms_are_involutions() -> None:
    """MOVE_MIRROR and MOVE_ROT180 are involutions; specials map to themselves."""
    ident = np.arange(len(MOVE_MIRROR))
    assert np.array_equal(MOVE_MIRROR[MOVE_MIRROR], ident)
    assert np.array_equal(MOVE_ROT180[MOVE_ROT180], ident)
    for special in range(4):
        assert MOVE_MIRROR[special] == special
        assert MOVE_ROT180[special] == special


def test_move_mirror_specific() -> None:
    """A concrete move mirrors to the geometrically correct square.

    Mirror (f, r) -> (8 - f, r). ``h2e2`` (files h=7, e=4) -> ``b2e2``
    (files 8-7=1='b', 8-4=4='e'), ranks unchanged.
    """
    tok = MoveTokenizer()
    m = tok.encode_move(Move.from_iccs("h2e2"))
    mirrored = int(MOVE_MIRROR[m])
    assert tok.id_to_move(mirrored) == Move.from_iccs("b2e2")


def test_move_rot180_specific() -> None:
    """Rotate-180 (f, r) -> (8 - f, 9 - r). ``h2e2`` -> ``b7e7``."""
    tok = MoveTokenizer()
    m = tok.encode_move(Move.from_iccs("h2e2"))
    rot = int(MOVE_ROT180[m])
    assert tok.id_to_move(rot) == Move.from_iccs("b7e7")


def test_board_mirror_decodes_to_legal_flipped_fen() -> None:
    """Mirroring the start position flips files and decodes to a legal FEN."""
    board_tok = BoardTokenizer()
    ids = np.array(board_tok.encode(STARTING_FEN), dtype=np.uint8)
    mirrored = ids[POS_MIRROR]
    decoded = board_tok.decode(mirrored.tolist())
    # The mirror of the (symmetric-except-cannons/horses) start position must be
    # a legal position.
    placement, side = decoded.split()
    full_fen = f"{placement} {side} - - 0 1"
    board = new_board(full_fen)  # raises if illegal
    assert board.turn is Color.RED
    # The start position's rows are all left-right palindromic, so a mirror
    # round-trips to the same FEN; the file-flip itself is checked on a
    # non-palindromic hand-built row in test_board_mirror_hand_built_row.


def test_board_mirror_hand_built_row() -> None:
    """A non-palindromic hand-built position mirrors to the file-flipped board."""
    board_tok = BoardTokenizer()
    # Red chariot on a0 (file 0), everything else empty; black general k on e9,
    # red general K on e0 (needed for a legal FEN). Red to move.
    fen = "4k4/9/9/9/9/9/9/9/9/R3K4 r - - 0 1"
    ids = np.array(board_tok.encode(fen), dtype=np.uint8)
    mirrored = ids[POS_MIRROR]
    decoded = board_tok.decode(mirrored.tolist())
    placement, side = decoded.split()
    # a0 (file 0) chariot -> i0 (file 8); e0 King stays (file 4 -> 4); e9 stays.
    assert placement == "4k4/9/9/9/9/9/9/9/9/4K3R"
    assert side == "r"
    new_board(f"{placement} {side} - - 0 1")  # legal


def _flags_off_count(tmp_path: Path) -> int:
    out = tmp_path / "ds_off"
    stats = build_board_shards([_game()], out, mirror=False, color_augment=False)
    return stats.num_samples


def test_augment_factor_and_values(tmp_path: Path) -> None:
    """Both flags on -> 4x samples; value array is identical across variants."""
    base = _flags_off_count(tmp_path)
    assert base == len(_MOVES)

    out = tmp_path / "ds_aug"
    stats = build_board_shards([_game()], out, mirror=True, color_augment=True)
    assert stats.augment_factor == 4
    assert stats.num_samples == 4 * base

    _boards, _moves, values = load_board_arrays(out)
    assert values.shape[0] == 4 * base
    # Values are stored per-game as [orig-block, mirror-block, color-block,
    # both-block], each block == the original per-ply value block.
    orig_block = values[:base]
    for k in range(4):
        block = values[k * base : (k + 1) * base]
        assert np.array_equal(block, orig_block)

    meta = json.loads((out / "board_meta.json").read_text())
    assert meta["augment"] == {"mirror": True, "color": True, "factor": 4}
    assert meta["schema"] == "board-ds-v1"


def test_augment_factor_mirror_only(tmp_path: Path) -> None:
    """Mirror only -> 2x; color only -> 2x."""
    out_m = tmp_path / "ds_mirror"
    sm = build_board_shards([_game()], out_m, mirror=True, color_augment=False)
    assert sm.augment_factor == 2
    assert sm.num_samples == 2 * len(_MOVES)

    out_c = tmp_path / "ds_color"
    sc = build_board_shards([_game()], out_c, mirror=False, color_augment=True)
    assert sc.augment_factor == 2
    assert sc.num_samples == 2 * len(_MOVES)


def test_augmented_boards_and_moves_are_legal(tmp_path: Path) -> None:
    """Every augmented board parses as a legal FEN and its move is legal in it."""
    out = tmp_path / "ds_legal"
    build_board_shards([_game()], out, mirror=True, color_augment=True)
    boards, moves, _values = load_board_arrays(out)

    board_tok = BoardTokenizer()
    move_tok = MoveTokenizer()

    def _normalise_side(s: str) -> str:
        parts = s.split()
        if len(parts) == 2 and parts[1] == "w":
            return f"{parts[0]} r"
        return s

    # Spot-check a handful of samples across the variant blocks.
    n = boards.shape[0]
    for idx in (0, n // 4, n // 2, 3 * n // 4, n - 1):
        decoded = _normalise_side(board_tok.decode(boards[idx].tolist()))
        placement, side = decoded.split()
        full_fen = f"{placement} {side} - - 0 1"
        board = new_board(full_fen)  # raises if the position is illegal
        move = move_tok.id_to_move(int(moves[idx]))
        assert move is not None
        assert board.is_legal(move), f"sample {idx}: move {move.iccs} illegal in {full_fen}"
