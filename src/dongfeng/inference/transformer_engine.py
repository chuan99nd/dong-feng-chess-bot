"""Neural engine: a :class:`~dongfeng.protocol.engine.Engine` backed by a transformer (M3).

:class:`TransformerEngine` is where the neural model plugs into the SAME universal
:class:`~dongfeng.protocol.engine.Engine` contract every other bot implements, so
it is a drop-in for the arena, CLI, UCCI adapter, and conformance harness.

Decoding pipeline:

1. **Load a checkpoint** — a trained :class:`~dongfeng.model.transformer.TransformerPolicy`
   (or a fresh random-init model if no checkpoint is given).
2. **Tokenize the history** — the moves played so far are encoded as
   ``[BOS] m1 m2 ...`` with the :class:`~dongfeng.tokenizer.move_tokenizer.MoveTokenizer`.
3. **Legal-move masking from core** — the policy logits are restricted to the legal
   moves reported by :func:`dongfeng.core.new_board`, so the engine can NEVER emit
   an illegal move even if the raw policy puts mass on one.
4. **Sample / argmax** — a move is chosen from the masked distribution
   (``Temperature`` / ``TopK`` via :meth:`set_option`; temperature ``0`` = argmax).

The ``torch`` import is deferred to construction so importing this module never
requires the optional ``model`` extra.
"""

from __future__ import annotations

from typing import Any

from ..core import STARTING_FEN, new_board
from ..core.types import Move
from ..protocol.engine import Analysis, Engine, EngineInfo, ScoredMove, SearchLimits
from ..tokenizer.move_tokenizer import MoveTokenizer

_INSTALL_HINT = "The neural engine needs the 'model' extra: uv sync --extra model"


class TransformerEngine(Engine):
    """Transformer-backed :class:`Engine` with legal-masked policy decoding."""

    def __init__(self, checkpoint: str | None = None, *, device: str = "cpu") -> None:
        try:
            import torch  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(_INSTALL_HINT) from exc
        self._torch = torch
        self._device = device
        self._tok = MoveTokenizer()
        self._temperature = 0.0  # 0 = argmax
        self._top_k = 0  # 0 = no top-k restriction
        self._rng = torch.Generator().manual_seed(0)

        from ..model.transformer import TransformerConfig, TransformerPolicy  # noqa: PLC0415

        if checkpoint is not None:
            self._model, self._extra = TransformerPolicy.load(checkpoint, map_location=device)
        else:
            # Random-init fallback (still emits legal moves) — useful for conformance.
            cfg = TransformerConfig(vocab_size=self._tok.vocab_size, n_layer=2, n_embd=64, n_head=2)
            self._model, self._extra = TransformerPolicy(cfg), {}
        self._model.to(device).eval()

        self._board = new_board(STARTING_FEN)
        self._history: list[Move] = []

    # -- Engine protocol ----------------------------------------------------

    def id(self) -> EngineInfo:
        step = self._extra.get("step")
        name = "Dong Feng Neural" + (f" (step {step})" if step is not None else "")
        return EngineInfo(
            name=name,
            author="Dong Feng contributors",
            options={"Temperature": "0.0", "TopK": "0", "Seed": "0"},
        )

    def new_game(self) -> None:
        self._board = new_board(STARTING_FEN)
        self._history = []

    def set_position(self, fen: str, moves: list[Move]) -> None:
        self._board = new_board(fen)
        for m in moves:
            self._board.push(m)
        self._history = list(moves)

    def _policy_logits(self) -> Any:
        torch = self._torch
        ids = [self._tok.BOS_ID, *(self._tok.encode_move(m) for m in self._history)]
        ids = ids[-self._model.config.block_size :]
        x = torch.tensor([ids], dtype=torch.long, device=self._device)
        with torch.no_grad():
            logits = self._model(x)[0, -1]  # [vocab]
        return logits

    def _masked_scores(self) -> tuple[list[Move], Any]:
        """Return ``(legal_moves, probabilities)`` over the legal moves (or ``[], None``)."""
        torch = self._torch
        legal = self._board.legal_moves()
        if not legal:
            return [], None
        logits = self._policy_logits()
        legal_ids = torch.tensor([self._tok.encode_move(m) for m in legal], device=self._device)
        legal_logits = logits[legal_ids]
        temp = self._temperature if self._temperature > 0 else 1.0
        probs = torch.softmax(legal_logits / temp, dim=-1)
        return legal, probs

    def bestmove(self, limits: SearchLimits) -> Move:
        legal, probs = self._masked_scores()
        if not legal:
            raise ValueError("no legal moves in the current position")
        torch = self._torch
        if self._temperature <= 0:
            idx = int(torch.argmax(probs).item())
        else:
            p = probs
            if 0 < self._top_k < len(legal):
                top = torch.topk(p, self._top_k)
                filtered = torch.zeros_like(p)
                filtered[top.indices] = top.values
                p = filtered / filtered.sum()
            idx = int(torch.multinomial(p, 1, generator=self._rng).item())
        return legal[idx]

    def analyze(self, limits: SearchLimits) -> Analysis:
        legal, probs = self._masked_scores()
        if not legal:
            return Analysis(moves=[])
        order = sorted(range(len(legal)), key=lambda i: float(probs[i]), reverse=True)
        scored = [ScoredMove(move=legal[i], policy_prob=float(probs[i])) for i in order]
        return Analysis(moves=scored)

    def set_option(self, name: str, value: str) -> None:
        key = name.lower()
        if key == "temperature":
            self._temperature = float(value)
        elif key == "topk":
            self._top_k = int(value)
        elif key == "seed":
            self._rng = self._torch.Generator().manual_seed(int(value))
        elif key == "checkpoint":
            from ..model.transformer import TransformerPolicy  # noqa: PLC0415

            self._model, self._extra = TransformerPolicy.load(value, map_location=self._device)
            self._model.to(self._device).eval()

    def stop(self) -> None:
        """No-op: single-forward-pass inference has nothing to interrupt."""
