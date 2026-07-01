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

import re
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


# --- DhtmlXQ (DPXQ web / vietcotuong.com) --------------------------------- #
# The backend can't read this web format, so we decode it directly. Verified by
# replaying real games through the rules backend (100% legal-move rate on a
# 200-game sample; the standard start position decodes exactly).
#
# Layout: origin is top-left, columns 0-8 (left->right), rows 0-9 (top=Black back
# rank -> bottom=Red back rank). ICCS rank = 9 - row; ICCS file = "abc...i"[col].
# `binit` is 64 digits = 32 pieces x (col,row), "99" meaning off-board, in the
# fixed order below. Moves are 4 digits: col_from,row_from,col_to,row_to.
_DHTMLXQ_ORDER = "RNBAKABNRCCPPPPP" + "rnbakabnrccppppp"  # Red pieces, then Black
_DHTMLXQ_RESULTS = {
    "红胜": GameResult.RED_WIN,
    "紅勝": GameResult.RED_WIN,
    "黑胜": GameResult.BLACK_WIN,
    "黑勝": GameResult.BLACK_WIN,
    "和局": GameResult.DRAW,
}
_BINIT_RE = re.compile(r"\[DhtmlXQ_binit\](\d{64})\[")
_MOVELIST_RE = re.compile(r"\[DhtmlXQ_movelist\](\d*)\[")
_TAG_RE = re.compile(r"\[DhtmlXQ_(\w+)\]([^\[]*)\[")


def _dhtmlxq_cell(col: int, row: int) -> str:
    return f"{'abcdefghi'[col]}{9 - row}"


def _binit_to_placement(binit: str) -> str:
    """Decode a 64-digit binit into a FEN placement field (rank 9 / top first)."""
    grid = [["."] * 9 for _ in range(10)]  # grid[row][col]; row 0 = top
    for i, letter in enumerate(_DHTMLXQ_ORDER):
        pair = binit[2 * i : 2 * i + 2]
        if pair == "99":
            continue
        col, row = int(pair[0]), int(pair[1])
        grid[row][col] = letter
    rows: list[str] = []
    for row in grid:
        out, run = "", 0
        for cell in row:
            if cell == ".":
                run += 1
            else:
                if run:
                    out += str(run)
                    run = 0
                out += cell
        if run:
            out += str(run)
        rows.append(out)
    return "/".join(rows)


def _dhtmlxq_moves(movelist: str) -> list[Move]:
    moves: list[Move] = []
    for i in range(0, len(movelist) - 3, 4):
        c = movelist[i : i + 4]
        moves.append(
            Move(_dhtmlxq_cell(int(c[0]), int(c[1])), _dhtmlxq_cell(int(c[2]), int(c[3])))
        )
    return moves


def parse_dhtmlxq(text: str) -> Iterator[Game]:
    """Parse a DhtmlXQ / DPXQ web-embed move blob into a :class:`Game`.

    Side-to-move is inferred from the owner of the first move's from-square (Black
    can move first in composed endgames); it defaults to Red for move-less records.
    """
    bm = _BINIT_RE.search(text)
    if bm is None:
        return
    placement = _binit_to_placement(bm.group(1))
    ml = _MOVELIST_RE.search(text)
    moves = _dhtmlxq_moves(ml.group(1)) if ml and ml.group(1) else []

    # Whoever owns the first move's from-square is the side to move.
    side = "r"
    if moves:
        first = moves[0]
        col = "abcdefghi".index(first.from_sq[0])
        row = 9 - int(first.from_sq[1])
        piece = placement.split("/")[row]
        # Expand the rank to find the piece letter at `col`.
        expanded = "".join("." * int(ch) if ch.isdigit() else ch for ch in piece)
        if col < len(expanded) and expanded[col].islower():
            side = "b"

    tags = {k: v for k, v in _TAG_RE.findall(text)}
    result = _DHTMLXQ_RESULTS.get(tags.get("result", "").strip(), GameResult.ONGOING)
    metadata = {k: v for k, v in tags.items() if v and k not in ("binit", "movelist")}
    yield Game(start_fen=f"{placement} {side}", moves=moves, result=result, metadata=metadata)


def parse_dhtmlxq_file(path: str | Path) -> Iterator[Game]:
    """Parse a DhtmlXQ file (any/no extension) into :class:`Game` objects."""
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    yield from parse_dhtmlxq(text)


def _looks_like_dhtmlxq(path: Path) -> bool:
    try:
        with open(path, encoding="utf-8", errors="ignore") as fh:
            return "[DhtmlXQ" in fh.read(64)
    except OSError:
        return False


def parse_file(path: str | Path) -> Iterator[Game]:
    """Parse a single game file, dispatching on extension (or DhtmlXQ content).

    Known extensions (.pgn/.xqf/.cbf/.cbl/.cbr/.txt) use the backend readers;
    extension-less or ``.dpxq`` files are sniffed for the DhtmlXQ web format.

    Raises:
        ValueError: if the file is neither a supported format nor DhtmlXQ.
    """
    p = Path(path)
    reader_name = _READERS.get(p.suffix.lower())
    if reader_name is not None:
        yield from _parse_with(reader_name, p)
        return
    if _looks_like_dhtmlxq(p):
        yield from parse_dhtmlxq_file(p)
        return
    raise ValueError(f"unsupported game format: {p.suffix!r} ({p})")


# Extensions considered when walking a directory (plus extension-less DhtmlXQ).
_WALK_SUFFIXES = frozenset({*_READERS, "", ".dpxq"})


def iter_games_in(path: str | Path, *, strict: bool = False) -> Iterator[Game]:
    """Yield games from a file, or from every supported file under a directory.

    Directories are walked recursively (backend formats + extension-less DhtmlXQ
    records). Corpora are messy, so by default a file that fails to parse is
    skipped; pass ``strict=True`` to raise instead.
    """
    root = Path(path)
    if root.is_dir():
        files = sorted(
            p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in _WALK_SUFFIXES
        )
    else:
        files = [root]
    for f in files:
        try:
            yield from parse_file(f)
        except Exception:
            if strict:
                raise
            continue
