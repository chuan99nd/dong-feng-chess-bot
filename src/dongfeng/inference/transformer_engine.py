"""Neural engine: a :class:`~dongfeng.protocol.engine.Engine` backed by a transformer (stub).

Roadmap milestone: **M3**.

:class:`TransformerEngine` is the point where the neural model plugs into the SAME
universal :class:`~dongfeng.protocol.engine.Engine` contract that every other bot
(Pikafish wrapper, random baseline, third-party engines) implements. Because it
conforms to that Protocol, it is a drop-in for any match runner, arena, CLI, or MCP
server, and it can be validated with
:func:`dongfeng.protocol.conformance.run_conformance`.

Planned behavior (pinned in M3):

1. **Load a checkpoint** — a trained :class:`~dongfeng.model.base.PolicyModel`
   (see :mod:`dongfeng.training`) is loaded once at construction.
2. **Tokenize the position** — the current FEN (+ move history) is encoded via the
   board / move tokenizers (:mod:`dongfeng.tokenizer`).
3. **Apply legal-move masking from core** — the model's policy logits are masked to
   the legal moves reported by :func:`dongfeng.core.new_board` (a Board's
   ``legal_moves()``), so the engine can never emit an illegal move even if the raw
   policy assigns mass to one.
4. **Sample / argmax** — a move is chosen from the masked distribution (temperature
   / top-k configurable via :meth:`set_option`); :meth:`analyze` additionally
   surfaces per-move policy priors and (if the model has a value head) win
   probabilities as :class:`~dongfeng.protocol.engine.ScoredMove` entries.

Until M3, every method raises ``NotImplementedError("planned: M3")``.
"""

from __future__ import annotations

from ..core.types import Move
from ..protocol.engine import Analysis, Engine, EngineInfo, SearchLimits


class TransformerEngine(Engine):
    """Transformer-backed :class:`Engine` (M3 stub).

    Implements the universal engine contract so the neural model is interchangeable
    with any other bot. All methods raise ``NotImplementedError`` until milestone
    M3; the signatures match :class:`dongfeng.protocol.engine.Engine` exactly.
    """

    def __init__(self, checkpoint: str | None = None) -> None:
        """Create the engine, planning to load ``checkpoint`` (a trained policy model).

        Args:
            checkpoint: Path to a trained model checkpoint to load at construction.

        Raises:
            NotImplementedError: always — planned for milestone M3.
        """
        raise NotImplementedError("planned: M3")

    def id(self) -> EngineInfo:
        """Return static engine identification.

        Raises:
            NotImplementedError: always — planned for milestone M3.
        """
        raise NotImplementedError("planned: M3")

    def new_game(self) -> None:
        """Reset per-game state.

        Raises:
            NotImplementedError: always — planned for milestone M3.
        """
        raise NotImplementedError("planned: M3")

    def set_position(self, fen: str, moves: list[Move]) -> None:
        """Set the root position to ``fen`` then apply ``moves`` in order.

        Raises:
            NotImplementedError: always — planned for milestone M3.
        """
        raise NotImplementedError("planned: M3")

    def analyze(self, limits: SearchLimits) -> Analysis:
        """Evaluate the current position and return scored candidate moves.

        The M3 implementation tokenizes the position, runs the model, masks to legal
        moves from core, and returns policy priors (and win probabilities if a value
        head is present).

        Raises:
            NotImplementedError: always — planned for milestone M3.
        """
        raise NotImplementedError("planned: M3")

    def bestmove(self, limits: SearchLimits) -> Move:
        """Return the chosen move for the current position.

        The M3 implementation samples (or takes the argmax of) the legal-masked
        policy.

        Raises:
            NotImplementedError: always — planned for milestone M3.
        """
        raise NotImplementedError("planned: M3")

    def set_option(self, name: str, value: str) -> None:
        """Set an engine option (e.g. sampling temperature, top-k) by name.

        Raises:
            NotImplementedError: always — planned for milestone M3.
        """
        raise NotImplementedError("planned: M3")

    def stop(self) -> None:
        """Request that any in-progress evaluation stop.

        Raises:
            NotImplementedError: always — planned for milestone M3.
        """
        raise NotImplementedError("planned: M3")
