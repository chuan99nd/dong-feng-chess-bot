"""Wrap an external Pikafish (or any UCI/UCCI) binary as a Dong Feng engine.

:class:`PikafishEngine` launches a strong external Xiangqi engine as a
subprocess and speaks its line protocol, exposing it behind the universal
:class:`~dongfeng.protocol.engine.Engine` contract. This lets a
tournament-grade engine (Pikafish, ~Elo 3950, NNUE) drop into any Dong Feng
match runner, arena, or the CLI, and serve as a distillation *teacher*.

Two dialects are supported, matching the two dominant Xiangqi engine protocols:

* **UCI** (default) — as spoken by Pikafish. Handshake ``uci`` -> ``uciok``;
  positions via ``position``, search via ``go``, reply ``bestmove``.
* **UCCI** — the native Xiangqi protocol (e.g. ElephantEye / Eleeye). Handshake
  ``ucci`` -> ``ucciok``; the command grammar is otherwise the same subset we
  use here.

Configuration (via :meth:`set_option`, or environment fallbacks):

* ``EnginePath`` — path to the engine binary. Falls back to the
  ``DONGFENG_PIKAFISH`` environment variable, then to ``pikafish`` on ``PATH``.
* ``Protocol`` — ``"uci"`` (default) or ``"ucci"``.
* ``EvalFile`` — path to the NNUE weights (Pikafish needs this). Forwarded as a
  ``setoption`` to the engine.
* Any other option name is forwarded verbatim as ``setoption name <N> value <V>``.

Dependency policy: this module is **stdlib-only** (``subprocess``, ``os``,
``shutil``). It imports nothing at module load that can fail, and it only touches
the binary when the engine is actually *used* — a missing/invalid binary raises a
clear :class:`EngineNotAvailableError` at first use, never at import.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass

from ..core import STARTING_FEN, Move
from ..protocol.engine import Analysis, EngineInfo, ScoredMove, SearchLimits


@dataclass(frozen=True)
class PikafishEval:
    """A static evaluation of a position, from the side-to-move's perspective.

    UCI ``score`` is mover-relative. Exactly one of ``cp`` (centipawns) / ``mate``
    (mate-in-N; sign = side to move) is set for a well-formed eval; both ``None``
    means no score was seen.
    """

    cp: int | None = None
    mate: int | None = None


# Environment variable consulted when no ``EnginePath`` option is set.
_ENV_PATH = "DONGFENG_PIKAFISH"
# Default binary name looked up on PATH as a last resort.
_DEFAULT_BINARY = "pikafish"
# How long (seconds) to wait for handshake / readyok / bestmove before giving up.
_HANDSHAKE_TIMEOUT_S = 10.0


class EngineNotAvailableError(RuntimeError):
    """Raised when the external engine binary cannot be located or launched."""


class PikafishEngine:
    """A conforming :class:`~dongfeng.protocol.engine.Engine` backed by a subprocess.

    The subprocess is started lazily on first use (any of :meth:`new_game`,
    :meth:`set_position`, :meth:`analyze`, :meth:`bestmove`). This keeps
    construction and import side-effect-free, so the class can be imported and
    inspected even where no engine binary is installed.
    """

    def __init__(
        self,
        engine_path: str | None = None,
        protocol: str = "uci",
    ) -> None:
        self._engine_path = engine_path
        self._protocol = protocol.lower()
        # Options queued before launch and (re)applied after each ``newgame``.
        self._pending_options: dict[str, str] = {}
        # Current root position + moves, so we can restate it on demand.
        self._root_fen: str = STARTING_FEN
        self._moves: list[Move] = []
        self._proc: subprocess.Popen[str] | None = None

    # -- identity -----------------------------------------------------------

    def id(self) -> EngineInfo:
        options: dict[str, str] = {"Protocol": self._protocol}
        if self._engine_path is not None:
            options["EnginePath"] = self._engine_path
        options.update(self._pending_options)
        return EngineInfo(name="Pikafish (external)", author="Pikafish authors", options=options)

    # -- lifecycle ----------------------------------------------------------

    def new_game(self) -> None:
        self._ensure_started()
        self._send("ucinewgame" if self._protocol == "uci" else "newgame")
        self._wait_ready()

    def set_position(self, fen: str, moves: list[Move]) -> None:
        self._root_fen = fen
        self._moves = list(moves)
        self._ensure_started()
        self._send_position()

    # -- search -------------------------------------------------------------

    def analyze(self, limits: SearchLimits) -> Analysis:
        self._ensure_started()
        self._send_position()
        best, ponder = self._go_and_collect(limits)
        _ = ponder
        moves = [ScoredMove(move=best)] if best is not None else []
        return Analysis(moves=moves, depth=limits.depth or 0, nodes=0, time_ms=0)

    def bestmove(self, limits: SearchLimits) -> Move:
        self._ensure_started()
        self._send_position()
        best, _ = self._go_and_collect(limits)
        if best is None:
            raise EngineNotAvailableError("engine returned no bestmove")
        return best

    # -- static evaluation (distillation teacher) ---------------------------

    def evaluate(
        self, fen: str, *, depth: int = 18, moves: list[Move] | None = None
    ) -> PikafishEval:
        """Return Pikafish's mover-relative score for a position (``go depth``).

        Parses the deepest ``info … score cp N`` / ``score mate N`` line before
        ``bestmove``. Used to generate value/state labels for distillation
        (Phase 4). ``score`` is relative to the side to move, matching the
        board model's mover-relative value target.
        """
        self._root_fen = fen
        self._moves = list(moves or [])
        self._ensure_started()
        self._send_position()
        self._send(f"go depth {depth}")
        cp: int | None = None
        mate: int | None = None
        for line in self._read_until("bestmove"):
            if not line.startswith("info") or "score" not in line:
                continue
            if "lowerbound" in line or "upperbound" in line:
                continue  # skip fail-high/low bounds; wait for the exact score
            c, m = self._parse_info_score(line)
            if c is not None or m is not None:
                cp, mate = c, m  # keep the latest (deepest) exact score
        return PikafishEval(cp=cp, mate=mate)

    @staticmethod
    def _parse_info_score(line: str) -> tuple[int | None, int | None]:
        """Extract ``(cp, mate)`` from a UCI ``info … score cp/mate N`` line."""
        toks = line.split()
        try:
            i = toks.index("score")
        except ValueError:
            return None, None
        if i + 2 >= len(toks):
            return None, None
        kind, val = toks[i + 1], toks[i + 2]
        try:
            if kind == "cp":
                return int(val), None
            if kind == "mate":
                return None, int(val)
        except ValueError:
            return None, None
        return None, None

    # -- options / control --------------------------------------------------

    def set_option(self, name: str, value: str) -> None:
        if name == "EnginePath":
            self._engine_path = value or None
            return
        if name == "Protocol":
            self._protocol = value.lower()
            return
        # Anything else is a genuine engine option: remember it and, if the
        # process is already up, forward it live.
        self._pending_options[name] = value
        if self._proc is not None:
            self._send(f"setoption name {name} value {value}")

    def stop(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            self._send("stop")

    def close(self) -> None:
        """Terminate the subprocess if running (idempotent)."""
        proc = self._proc
        if proc is None:
            return
        self._proc = None
        if proc.poll() is None:
            try:
                self._raw_send(proc, "quit")
                proc.wait(timeout=2.0)
            except (OSError, subprocess.TimeoutExpired):
                proc.kill()

    def __del__(self) -> None:  # pragma: no cover - best-effort cleanup
        with contextlib.suppress(Exception):
            self.close()

    # -- process management -------------------------------------------------

    def _resolve_binary(self) -> str:
        """Return the engine binary path, or raise if it cannot be found."""
        candidate = self._engine_path or os.environ.get(_ENV_PATH) or _DEFAULT_BINARY
        resolved = shutil.which(candidate) or (candidate if os.path.isfile(candidate) else None)
        if resolved is None:
            raise EngineNotAvailableError(
                f"could not find engine binary {candidate!r}. Set it via "
                f'set_option("EnginePath", "/path/to/pikafish") or the '
                f"{_ENV_PATH} environment variable."
            )
        return resolved

    def _ensure_started(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            return
        binary = self._resolve_binary()
        try:
            self._proc = subprocess.Popen(  # noqa: S603 - path is user-provided by design
                [binary],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1,
            )
        except OSError as exc:
            raise EngineNotAvailableError(f"failed to launch {binary!r}: {exc}") from exc
        self._handshake()
        # Apply any queued options, then sync.
        for name, value in self._pending_options.items():
            self._send(f"setoption name {name} value {value}")
        self._wait_ready()

    def _handshake(self) -> None:
        handshake = "uci" if self._protocol == "uci" else "ucci"
        expected = "uciok" if self._protocol == "uci" else "ucciok"
        self._send(handshake)
        for line in self._read_until(expected):
            if line == expected:
                return
        raise EngineNotAvailableError(f"engine did not answer {expected!r} to {handshake!r}")

    def _wait_ready(self) -> None:
        self._send("isready")
        for line in self._read_until("readyok"):
            if line == "readyok":
                return
        raise EngineNotAvailableError("engine did not answer 'readyok' to 'isready'")

    def _send_position(self) -> None:
        parts = ["position"]
        if self._root_fen == STARTING_FEN:
            parts.append("startpos")
        else:
            parts += ["fen", self._root_fen]
        if self._moves:
            parts.append("moves")
            parts += [m.iccs for m in self._moves]
        self._send(" ".join(parts))

    def _go_and_collect(self, limits: SearchLimits) -> tuple[Move | None, Move | None]:
        """Send ``go`` with ``limits`` and parse the ``bestmove`` reply.

        Returns ``(bestmove, ponder)`` as :class:`Move` objects (either may be
        ``None`` if the engine reports ``bestmove (none)``).
        """
        self._send(self._go_command(limits))
        for line in self._read_until("bestmove"):
            if line.startswith("bestmove"):
                return self._parse_bestmove(line)
        raise EngineNotAvailableError("engine did not return a 'bestmove'")

    @staticmethod
    def _go_command(limits: SearchLimits) -> str:
        parts = ["go"]
        if limits.movetime_ms is not None:
            parts += ["movetime", str(limits.movetime_ms)]
        if limits.depth is not None:
            parts += ["depth", str(limits.depth)]
        if limits.nodes is not None:
            parts += ["nodes", str(limits.nodes)]
        if limits.wtime_ms is not None:
            parts += ["wtime", str(limits.wtime_ms)]
        if limits.btime_ms is not None:
            parts += ["btime", str(limits.btime_ms)]
        # With no explicit limit, ask for a modest fixed-depth search so ``go``
        # terminates rather than pondering forever.
        if len(parts) == 1:
            parts += ["depth", "12"]
        return " ".join(parts)

    @staticmethod
    def _parse_bestmove(line: str) -> tuple[Move | None, Move | None]:
        tokens = line.split()
        best: Move | None = None
        ponder: Move | None = None
        if len(tokens) >= 2 and tokens[1] not in ("(none)", "none", "0000"):
            best = Move.from_iccs(tokens[1])
        if len(tokens) >= 4 and tokens[2] == "ponder" and tokens[3] not in ("(none)", "none"):
            ponder = Move.from_iccs(tokens[3])
        return best, ponder

    # -- low-level I/O ------------------------------------------------------

    def _send(self, command: str) -> None:
        proc = self._proc
        if proc is None:
            raise EngineNotAvailableError("engine process is not running")
        self._raw_send(proc, command)

    @staticmethod
    def _raw_send(proc: subprocess.Popen[str], command: str) -> None:
        if proc.stdin is None:
            raise EngineNotAvailableError("engine stdin is unavailable")
        proc.stdin.write(command + "\n")
        proc.stdin.flush()

    def _read_until(self, sentinel: str) -> Iterator[str]:
        """Yield engine output lines up to and including one starting with ``sentinel``.

        Stops on the sentinel or on EOF / process death. Raises if the process
        exits before the sentinel is seen.
        """
        proc = self._proc
        if proc is None or proc.stdout is None:
            raise EngineNotAvailableError("engine stdout is unavailable")
        while True:
            raw = proc.stdout.readline()
            if raw == "":  # EOF / process exited
                raise EngineNotAvailableError(
                    f"engine closed the connection before emitting {sentinel!r}"
                )
            line = raw.strip()
            if not line:
                continue
            yield line
            if line == sentinel or line.startswith(sentinel + " "):
                return
