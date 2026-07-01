"""Ingestion: parse raw Xiangqi game records into :class:`~dongfeng.data.base.Game`.

Roadmap milestone: **M1** (landed).

Each supported format gets a parser that yields :class:`Game` objects, so
downstream code (dataset sharding, filtering) is format-agnostic. Parsing is
delegated to the ``cchess`` backend (walker8088), which reads the formats found in
the wild and exposes each game as ``(FEN, ICCS-move)`` pairs:

* **PGN-for-Xiangqi** (``.pgn``) — the interchange format used by the large
  published corpora (Xiangqi-R1 ~183k pro games; the ~15M TianTian records).
* **XQF** (``.xqf``) — the de-facto DPXQ (dpxq.com) / XiangQi Studio binary format.
* **CBF / CBL / CBR** (``.cbf`` / ``.cbl`` / ``.cbr``) — the CCBridge family.
* **TXT** (``.txt``) — the plain move-list export.

``DhtmlXQ`` (the DPXQ web/UBB embed) has no backend reader and is left as a
scrape-and-convert task.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from ..core.types import GameResult, Move
from .base import Game

# Map cchess PGN result tags to our GameResult.
_RESULT_MAP = {
    "1-0": GameResult.RED_WIN,
    "0-1": GameResult.BLACK_WIN,
    "1/2-1/2": GameResult.DRAW,
}

#: File extensions this module can parse, mapped to their backend reader name.
_READERS: dict[str, str] = {
    ".pgn": "read_from_pgn",
    ".xqf": "read_from_xqf",
    ".cbf": "read_from_cbf",
    ".cbl": "read_from_cbl",
    ".cbr": "read_from_cbr",
    ".txt": "read_from_txt",
}

_INSTALL_HINT = (
    "The 'cchess' rules library is required for ingestion. "
    "Install it with: uv pip install 'cchess>=1.25,<2'"
)


def _require_cchess() -> Any:
    try:
        import cchess  # noqa: PLC0415  (deferred import by design)
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(_INSTALL_HINT) from exc
    return cchess


def _result_from_info(info: dict[str, Any]) -> GameResult:
    return _RESULT_MAP.get(str(info.get("result", "")).strip(), GameResult.ONGOING)


def _game_from_cchess(cg: Any) -> Game:
    """Convert a ``cchess`` game object into our :class:`Game` (main line only)."""
    variations = cg.dump_iccs_moves()  # list of variations; [0] is the main line
    mainline = variations[0] if variations else []
    moves = [Move.from_iccs(s) for s in mainline]
    info = dict(getattr(cg, "info", {}) or {})
    return Game(
        start_fen=cg.init_board.to_fen(),
        moves=moves,
        result=_result_from_info(info),
        metadata={str(k): str(v) for k, v in info.items()},
    )


def _parse_with(reader_name: str, path: str | Path) -> Iterator[Game]:
    cchess = _require_cchess()
    reader = getattr(cchess, reader_name)
    cg = reader(str(path))
    if cg is not None:
        yield _game_from_cchess(cg)


def parse_pgn(path: str | Path) -> Iterator[Game]:
    """Parse a PGN-for-Xiangqi file into :class:`Game` objects."""
    yield from _parse_with("read_from_pgn", path)


def parse_xqf(path: str | Path) -> Iterator[Game]:
    """Parse a DPXQ / XiangQi Studio XQF binary record into :class:`Game` objects."""
    yield from _parse_with("read_from_xqf", path)


def parse_cbf(path: str | Path) -> Iterator[Game]:
    """Parse a CCBridge CBF record into :class:`Game` objects."""
    yield from _parse_with("read_from_cbf", path)


def parse_cbl(path: str | Path) -> Iterator[Game]:
    """Parse a CCBridge CBL library into :class:`Game` objects."""
    yield from _parse_with("read_from_cbl", path)


def parse_cbr(path: str | Path) -> Iterator[Game]:
    """Parse a CCBridge CBR record into :class:`Game` objects."""
    yield from _parse_with("read_from_cbr", path)


def parse_txt(path: str | Path) -> Iterator[Game]:
    """Parse a plain move-list TXT export into :class:`Game` objects."""
    yield from _parse_with("read_from_txt", path)


def parse_dhtmlxq(text: str) -> Iterator[Game]:
    """Parse a DPXQ DhtmlXQ / UBB web-embed move blob into :class:`Game` objects.

    The backend has no reader for this web format; scrape it and convert the move
    codes to ICCS, or save the game as PGN/XQF and use those parsers instead.

    Raises:
        NotImplementedError: no backend reader exists for this format.
    """
    raise NotImplementedError(
        "DhtmlXQ has no cchess reader; scrape+convert to ICCS or re-export as PGN/XQF"
    )


def parse_file(path: str | Path) -> Iterator[Game]:
    """Parse a single game file, dispatching on its extension.

    Raises:
        ValueError: if the file extension is not a supported format.
    """
    suffix = Path(path).suffix.lower()
    reader_name = _READERS.get(suffix)
    if reader_name is None:
        raise ValueError(f"unsupported game format: {suffix!r} ({path})")
    yield from _parse_with(reader_name, path)


def iter_games_in(path: str | Path, *, strict: bool = False) -> Iterator[Game]:
    """Yield games from a file, or from every supported file under a directory.

    Directories are walked recursively. Corpora are messy, so by default a file
    that fails to parse is skipped; pass ``strict=True`` to raise instead.
    """
    root = Path(path)
    if root.is_dir():
        files = sorted(p for p in root.rglob("*") if p.suffix.lower() in _READERS)
    else:
        files = [root]
    for f in files:
        try:
            yield from parse_file(f)
        except Exception:
            if strict:
                raise
            continue
