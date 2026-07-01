"""Text line-protocol adapter that exposes any :class:`Engine` as a UCCI / UCI bot.

Dong Feng speaks two closely-related text dialects over a line-oriented I/O
stream so that a conforming :class:`~dongfeng.protocol.engine.Engine` can be
driven by real GUIs and tournament managers:

* **UCCI** (Universal Chinese Chess Interface) — the native Xiangqi engine
  protocol. Handshake command is ``ucci``; the engine answers ``ucciok``.
* **UCI** (as spoken by Pikafish for Xiangqi) — the Stockfish-derived dialect.
  Handshake command is ``uci``; the engine answers ``uciok``.

The two dialects are near-identical at the command level (both use
``position``/``go``/``bestmove``/``isready``/``setoption``/``stop``/``quit``),
so a single dispatcher serves both; only the handshake reply differs. The
adapter remembers which handshake word it saw and answers ``ucciok`` vs
``uciok`` to match.

Design goals:

* **Dependency-free** — stdlib only. Board / rules logic lives behind the
  injected :class:`Engine`; this module only parses and formats text.
* **Testable** — :meth:`ProtocolAdapter.run` takes arbitrary ``stdin`` / ``stdout``
  streams (any file-like object with ``readline`` / ``write``), so tests feed a
  scripted :class:`io.StringIO` and read back the output.

Moves on the wire are ICCS 4-char strings (e.g. ``h2e2``); positions are Xiangqi
FEN. Xiangqi has no promotion, so a move token is always exactly 4 characters.
"""

from __future__ import annotations

from typing import TextIO

from ..core.fen import STARTING_FEN
from ..core.types import Move
from .engine import Engine, SearchLimits


class ProtocolAdapter:
    """Bridge a line-oriented UCCI/UCI stream to an :class:`Engine`.

    Construct with the engine to expose, then call :meth:`run` with the input and
    output streams to service commands until ``quit`` (or EOF).

    Example (in-memory, for tests)::

        import io
        adapter = ProtocolAdapter(RandomEngine())
        out = io.StringIO()
        adapter.run(io.StringIO("uci\\nisready\\nposition startpos\\ngo\\nquit\\n"), out)
        print(out.getvalue())
    """

    def __init__(self, engine: Engine) -> None:
        self._engine = engine
        # The FEN of the root position for the current ``position`` command.
        self._root_fen: str = STARTING_FEN
        # Moves applied on top of the root, in order.
        self._moves: list[Move] = []
        # Which handshake dialect the peer used: "ucci" or "uci". Defaults to
        # UCCI (the native Xiangqi protocol) until a handshake is seen.
        self._dialect: str = "ucci"

    # -- public entry point -------------------------------------------------

    def run(self, stdin: TextIO, stdout: TextIO) -> None:
        """Service commands read from ``stdin``, writing replies to ``stdout``.

        Reads one command per line until ``quit`` is received or the input
        reaches EOF. Each reply line is written with a trailing newline and the
        stream is flushed (when the stream supports ``flush``) so an interactive
        peer sees replies immediately.
        """
        while True:
            line = stdin.readline()
            if line == "":  # EOF
                break
            command = line.strip()
            if not command:
                continue
            should_quit = self._dispatch(command, stdout)
            if should_quit:
                break

    # -- command dispatch ---------------------------------------------------

    def _dispatch(self, command: str, stdout: TextIO) -> bool:
        """Handle one command line. Return ``True`` iff the loop should stop."""
        tokens = command.split()
        verb = tokens[0]
        args = tokens[1:]

        if verb in ("ucci", "uci"):
            self._handle_handshake(verb, stdout)
        elif verb == "isready":
            self._emit(stdout, "readyok")
        elif verb in ("ucinewgame", "newgame"):
            self._engine.new_game()
        elif verb == "position":
            self._handle_position(args)
        elif verb == "go":
            self._handle_go(args, stdout)
        elif verb == "setoption":
            self._handle_setoption(args)
        elif verb == "stop":
            self._engine.stop()
        elif verb in ("quit", "bye"):
            return True
        # Unknown commands are ignored, as UCI/UCCI both mandate.
        return False

    # -- handlers -----------------------------------------------------------

    def _handle_handshake(self, verb: str, stdout: TextIO) -> None:
        """Reply to a ``ucci``/``uci`` handshake with id lines and the ok line."""
        self._dialect = verb
        info = self._engine.id()
        self._emit(stdout, f"id name {info.name}")
        self._emit(stdout, f"id author {info.author}")
        # Advertise declared options (best-effort; harmless if a GUI ignores them).
        for opt_name, opt_value in info.options.items():
            self._emit(
                stdout,
                f"option name {opt_name} type string default {opt_value}",
            )
        self._emit(stdout, "ucciok" if verb == "ucci" else "uciok")

    def _handle_position(self, args: list[str]) -> None:
        """Parse ``position (startpos | fen <FEN...>) [moves m1 m2 ...]``."""
        if not args:
            return

        moves_ix = args.index("moves") if "moves" in args else len(args)
        head = args[:moves_ix]
        move_tokens = args[moves_ix + 1 :] if moves_ix < len(args) else []

        if head and head[0] == "startpos":
            self._root_fen = STARTING_FEN
        elif head and head[0] == "fen":
            # The FEN may be 2 or 6 space-separated fields; take everything up to
            # the (already-stripped) ``moves`` keyword.
            fen = " ".join(head[1:])
            if fen:
                self._root_fen = fen
        # else: malformed ``position`` — keep the previous root.

        self._moves = [Move.from_iccs(tok) for tok in move_tokens]
        self._engine.set_position(self._root_fen, self._moves)

    def _handle_go(self, args: list[str], stdout: TextIO) -> None:
        """Parse ``go`` flags into :class:`SearchLimits`, then emit ``bestmove``."""
        limits = self._parse_go(args)
        move = self._engine.bestmove(limits)
        self._emit(stdout, f"bestmove {move.iccs}")

    def _handle_setoption(self, args: list[str]) -> None:
        """Parse ``setoption name <NAME> [value <VALUE...>]`` and forward it.

        Both dialects use the ``name ... value ...`` shape. The name may contain
        spaces; everything after ``value`` (which may also contain spaces) is the
        value. If ``value`` is omitted, an empty string is passed.
        """
        if not args or args[0] != "name":
            return
        try:
            value_ix = args.index("value")
        except ValueError:
            name = " ".join(args[1:])
            value = ""
        else:
            name = " ".join(args[1:value_ix])
            value = " ".join(args[value_ix + 1 :])
        if name:
            self._engine.set_option(name, value)

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _parse_go(args: list[str]) -> SearchLimits:
        """Translate ``go`` flags into a :class:`SearchLimits`.

        Recognized flags (values are integers): ``movetime`` (ms), ``depth``
        (plies), ``nodes``, ``wtime``/``btime`` (ms). ``infinite`` and unknown
        flags are ignored (the engine searches at its own discretion).
        """
        limits = SearchLimits()
        i = 0
        while i < len(args):
            flag = args[i]
            if flag in ("movetime", "depth", "nodes", "wtime", "btime") and i + 1 < len(args):
                raw = args[i + 1]
                try:
                    value = int(raw)
                except ValueError:
                    i += 2
                    continue
                if flag == "movetime":
                    limits.movetime_ms = value
                elif flag == "depth":
                    limits.depth = value
                elif flag == "nodes":
                    limits.nodes = value
                elif flag == "wtime":
                    limits.wtime_ms = value
                elif flag == "btime":
                    limits.btime_ms = value
                i += 2
            else:
                i += 1
        return limits

    @staticmethod
    def _emit(stdout: TextIO, line: str) -> None:
        """Write one reply line and flush if the stream supports it."""
        stdout.write(line + "\n")
        flush = getattr(stdout, "flush", None)
        if callable(flush):
            flush()
