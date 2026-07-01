"""Dataset: turn parsed games into per-ply samples and sharded token streams.

Roadmap milestone: **M1** (landed).

Two products, from the same parsed games:

* :func:`iter_samples` — explode games into per-ply :class:`~dongfeng.data.base.Sample`
  objects ``(FEN, move, turn, result)``. Optionally keep only the *winning* side's
  moves in decisive games (both sides in draws), the standard recipe for not
  training on the losing side's mistakes. This is the board-conditioned /
  supervised view.

* :func:`build_shards` — the flagship autoregressive view: encode each game with
  the :class:`~dongfeng.tokenizer.move_tokenizer.MoveTokenizer` as
  ``[BOS] m1 ... mN [EOS]`` and concatenate into fixed-size ``uint16`` binary
  shards (nanoGPT-style), plus a ``dataset_meta.json`` with the counts. Whole games
  are kept (both sides) so the alternating move sequence stays intact.

Shards are written with the stdlib :mod:`array` module (no numpy dependency); ids
fit in ``uint16`` since the move vocabulary is ~2086.
"""

from __future__ import annotations

import json
from array import array
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..core import GameResult, new_board
from ..core.types import Color
from ..tokenizer.move_tokenizer import MoveTokenizer
from .base import Game, Sample

_WIN_COLOR = {GameResult.RED_WIN: Color.RED, GameResult.BLACK_WIN: Color.BLACK}


def iter_samples(games: Iterable[Game], *, winning_side_only: bool = False) -> Iterator[Sample]:
    """Explode parsed games into per-ply :class:`Sample` objects.

    Args:
        games: Parsed games to convert.
        winning_side_only: In decisive games, yield only the winner's moves (draws
            always yield both sides). Skips the loser's mistakes for cleaner BC.

    A game is replayed on a real board; if a move is illegal (corrupt record), the
    rest of that game is skipped.
    """
    for g in games:
        board = new_board(g.start_fen)
        keep_color = _WIN_COLOR.get(g.result) if winning_side_only else None
        for move in g.moves:
            if not board.is_legal(move):
                break  # corrupt record: stop replaying this game
            turn = board.turn
            if keep_color is None or turn == keep_color:
                yield Sample(fen=board.fen(), move=move, turn=turn, result=g.result)
            board.push(move)


@dataclass(slots=True)
class BuildStats:
    """Summary of a :func:`build_shards` run (also written to ``dataset_meta.json``)."""

    num_games: int = 0
    num_samples: int = 0  # plies actually encoded (== (FEN, move) pairs)
    num_tokens: int = 0  # total token ids written (incl. BOS/EOS)
    num_shards: int = 0
    skipped_games: int = 0  # games dropped as empty or containing an illegal move
    tokenizer: str = MoveTokenizer.id
    vocab_size: int = 0
    shard_paths: list[str] = field(default_factory=list)


def build_shards(
    games: Iterable[Game],
    out_dir: str | Path,
    *,
    tokenizer: MoveTokenizer | None = None,
    shard_size: int = 1_048_576,
    created: str | None = None,
) -> BuildStats:
    """Encode games into ``uint16`` autoregressive token shards on disk.

    Args:
        games: Parsed games (from :mod:`dongfeng.data.ingest`).
        out_dir: Directory to write ``shard_XXXXX.bin`` and ``dataset_meta.json`` into.
        tokenizer: Move tokenizer to use (defaults to a fresh :class:`MoveTokenizer`).
        shard_size: Target token ids per shard file.
        created: Optional ISO-8601 timestamp recorded in the metadata.

    Returns:
        A :class:`BuildStats` with the counts and shard paths.
    """
    tok = tokenizer or MoveTokenizer()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    stats = BuildStats(tokenizer=tok.id, vocab_size=tok.vocab_size)
    buffer: array[int] = array("H")  # uint16

    def _flush() -> None:
        if not buffer:
            return
        shard_path = out / f"shard_{stats.num_shards:05d}.bin"
        with open(shard_path, "wb") as fh:
            buffer.tofile(fh)
        stats.shard_paths.append(str(shard_path))
        stats.num_shards += 1
        del buffer[:]

    for g in games:
        # Validate the game is fully replayable before committing its tokens.
        board = new_board(g.start_fen)
        replayable: list = []
        for move in g.moves:
            if not board.is_legal(move):
                break
            replayable.append(move)
            board.push(move)
        if not replayable:
            stats.skipped_games += 1
            continue
        ids = tok.encode_game(replayable, add_special=True)
        buffer.extend(ids)
        stats.num_games += 1
        stats.num_samples += len(replayable)
        stats.num_tokens += len(ids)
        if len(buffer) >= shard_size:
            _flush()
    _flush()

    meta = asdict(stats)
    meta["created"] = created
    meta["shard_size"] = shard_size
    with open(out / "dataset_meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    return stats
