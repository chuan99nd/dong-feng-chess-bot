"""Board-state dataset pipeline — builds per-ply binary shards for the board model.

Roadmap milestone: **M3.5 / WP1** (board-state model).

Each *sample* is one ply of a game:
  * **board** — the 91-token ``board-v1`` encoding of the position *before* the move.
  * **move** — the ``move-v1`` token id of the move played.
  * **value** — game-outcome from the side-to-move's perspective:
    ``+1`` (win), ``-1`` (loss), ``0`` (draw), ``127`` (mask / unknown).

Three parallel shard files share the same sample index ``N``:
  ``boards_XXXXX.bin``  — ``uint8``, shape ``N × 91``, C-order.
  ``moves_XXXXX.bin``   — ``uint16``, length ``N``.
  ``values_XXXXX.bin``  — ``int8``,  length ``N``.

A ``board_meta.json`` file holds the schema tag and counts (§1.3).
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..core import GameResult, new_board
from ..tokenizer.board_tokenizer import BoardTokenizer
from ..tokenizer.move_tokenizer import MoveTokenizer
from .base import Game

# Sentinel for unknown/ongoing game result.
_MASK_VALUE: int = 127

# Map (result, turn) → value target.
# Built lazily to avoid circular references at import time — but both enums are
# available so we can build it inline here.
from ..core.types import Color  # noqa: E402 (needed after enums imported)

_RESULT_TURN_TO_VALUE: dict[tuple[GameResult, Color], int] = {
    (GameResult.RED_WIN, Color.RED): 1,
    (GameResult.RED_WIN, Color.BLACK): -1,
    (GameResult.BLACK_WIN, Color.BLACK): 1,
    (GameResult.BLACK_WIN, Color.RED): -1,
    (GameResult.DRAW, Color.RED): 0,
    (GameResult.DRAW, Color.BLACK): 0,
}


def _value_target(result: GameResult, turn: Color) -> int:
    """Return the int8 value target for a (result, turn) pair."""
    return _RESULT_TURN_TO_VALUE.get((result, turn), _MASK_VALUE)


@dataclass
class BoardBuildStats:
    """Summary returned (and written) by :func:`build_board_shards`."""

    num_samples: int = 0
    num_games: int = 0
    skipped_games: int = 0
    shards: list[str] = field(default_factory=list)
    out_dir: str = ""


def build_board_shards(
    games: Iterable[Game],
    out_dir: str | Path,
    *,
    shard_size: int = 1_000_000,
    created: str | None = None,
) -> BoardBuildStats:
    """Encode games into per-ply board-state binary shards on disk.

    Args:
        games: Parsed games to encode (same type as :func:`~dongfeng.data.iter_samples`).
        out_dir: Directory to write ``boards_XXXXX.bin``, ``moves_XXXXX.bin``,
            ``values_XXXXX.bin``, and ``board_meta.json`` into.
        shard_size: Maximum number of samples per shard file.
        created: Optional ISO-8601 timestamp recorded in the metadata.

    Returns:
        A :class:`BoardBuildStats` with the counts and shard basenames.

    Games (or individual plies) that fail to tokenize are silently skipped and
    counted in :attr:`BoardBuildStats.skipped_games`.
    """
    board_tok = BoardTokenizer()
    move_tok = MoveTokenizer()

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    stats = BoardBuildStats(out_dir=str(out))

    # In-memory buffers for the current shard (pre-allocated in chunks).
    _boards_buf: list[list[int]] = []
    _moves_buf: list[int] = []
    _values_buf: list[int] = []
    _shard_idx: int = 0

    def _flush(buf_boards: list[list[int]], buf_moves: list[int], buf_values: list[int]) -> None:
        nonlocal _shard_idx
        if not buf_boards:
            return
        name_stem = f"{_shard_idx:05d}"
        n = len(buf_boards)

        boards_arr = np.array(buf_boards, dtype=np.uint8)  # shape (n, 91)
        moves_arr = np.array(buf_moves, dtype=np.uint16)
        values_arr = np.array(buf_values, dtype=np.int8)

        boards_path = out / f"boards_{name_stem}.bin"
        moves_path = out / f"moves_{name_stem}.bin"
        values_path = out / f"values_{name_stem}.bin"

        boards_arr.tofile(boards_path)
        moves_arr.tofile(moves_path)
        values_arr.tofile(values_path)

        # Record just the filename (not full path) in shards list, consistent with §1.3.
        stats.shards.append(name_stem)
        _shard_idx += 1

        # Clear buffers in-place by reassigning (caller must rebind).
        del buf_boards[:]
        del buf_moves[:]
        del buf_values[:]

        _ = n  # used above; suppress lint

    boards_buf: list[list[int]] = _boards_buf
    moves_buf: list[int] = _moves_buf
    values_buf: list[int] = _values_buf

    for g in games:
        board = new_board(g.start_fen)
        game_boards: list[list[int]] = []
        game_moves: list[int] = []
        game_values: list[int] = []
        game_ok = True

        for move in g.moves:
            if not board.is_legal(move):
                game_ok = False
                break
            turn = board.turn
            fen = board.fen()
            try:
                board_ids = board_tok.encode(fen)
                move_id = move_tok.encode_move(move)
            except Exception:  # noqa: BLE001
                game_ok = False
                break

            value = _value_target(g.result, turn)
            game_boards.append(board_ids)
            game_moves.append(move_id)
            game_values.append(value)
            board.push(move)

        if not game_ok or not game_boards:
            stats.skipped_games += 1
            continue

        # Commit this game's plies into the shared buffers.
        boards_buf.extend(game_boards)
        moves_buf.extend(game_moves)
        values_buf.extend(game_values)
        stats.num_games += 1
        stats.num_samples += len(game_boards)

        # Flush when the buffer is full.
        if len(boards_buf) >= shard_size:
            _flush(boards_buf, moves_buf, values_buf)

    # Flush remaining samples.
    _flush(boards_buf, moves_buf, values_buf)

    meta: dict = {
        "schema": "board-ds-v1",
        "num_samples": stats.num_samples,
        "num_games": stats.num_games,
        "skipped_games": stats.skipped_games,
        "tokenizer": BoardTokenizer.id,
        "move_tokenizer": MoveTokenizer.id,
        "shards": stats.shards,
        "created": created,
    }
    with open(out / "board_meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    return stats


def load_board_arrays(
    data_dir: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load all board shards from *data_dir* into memmap-friendly numpy arrays.

    Returns:
        ``(boards, moves, values)`` where:
          * ``boards`` — ``uint8`` array of shape ``(N, 91)``.
          * ``moves``  — ``uint16`` array of shape ``(N,)``.
          * ``values`` — ``int8``  array of shape ``(N,)``.

    When the dataset fits in RAM, the arrays are read directly; otherwise the
    caller can switch to :func:`numpy.memmap` themselves using the shard paths
    in ``board_meta.json``.  This loader concatenates shards for convenience
    but uses ``np.memmap`` per shard to keep peak memory low.
    """
    data_dir = Path(data_dir)
    meta_path = data_dir / "board_meta.json"
    with open(meta_path, encoding="utf-8") as fh:
        meta = json.load(fh)

    num_samples: int = meta["num_samples"]
    shard_names: list[str] = meta["shards"]

    if num_samples == 0 or not shard_names:
        boards = np.empty((0, 91), dtype=np.uint8)
        moves = np.empty((0,), dtype=np.uint16)
        values = np.empty((0,), dtype=np.int8)
        return boards, moves, values

    boards_parts: list[np.ndarray] = []
    moves_parts: list[np.ndarray] = []
    values_parts: list[np.ndarray] = []

    for stem in shard_names:
        boards_path = data_dir / f"boards_{stem}.bin"
        moves_path = data_dir / f"moves_{stem}.bin"
        values_path = data_dir / f"values_{stem}.bin"

        # Use memmap to avoid loading everything at once during iteration.
        raw_boards = np.memmap(boards_path, dtype=np.uint8, mode="r")
        n = raw_boards.size // 91
        boards_parts.append(raw_boards[: n * 91].reshape(n, 91))
        moves_parts.append(np.memmap(moves_path, dtype=np.uint16, mode="r")[:n])
        values_parts.append(np.memmap(values_path, dtype=np.int8, mode="r")[:n])

    boards = np.concatenate(boards_parts, axis=0)
    moves = np.concatenate(moves_parts, axis=0)
    values = np.concatenate(values_parts, axis=0)
    return boards, moves, values
