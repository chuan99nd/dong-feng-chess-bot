"""Board-state transformer engine — Engine protocol backed by :class:`BoardTransformer`.

:class:`BoardTransformerEngine` encodes the current board position via
:class:`~dongfeng.tokenizer.board_tokenizer.BoardTokenizer` (91 tokens), runs a
forward pass through :class:`~dongfeng.model.board_transformer.BoardTransformer`,
applies a **legal-move mask** over the move-v1 vocabulary, and returns the chosen
move plus an analysis annotated with both ``policy_prob`` and ``win_prob``.

Decoding pipeline
-----------------
1. Encode the current FEN → ``[B=1, 91]`` LongTensor via :class:`BoardTokenizer`.
2. Run ``model.forward(boards)`` → ``(policy_logits[1, 2554], value[1])``.
3. Build the legal set: ``board.legal_moves()`` → move-v1 ids.
4. Mask all illegal logits to ``-inf``; softmax over legal ids.
5. Pick move: argmax if ``Temperature ≤ 0``, else temperature / top-k sampling.
6. ``win_prob = (value + 1) / 2`` maps tanh (−1, 1) → [0, 1] for the side-to-move.

The ``torch`` import is deferred to construction so importing this module never
requires the optional ``model`` extra.
"""

from __future__ import annotations

from typing import Any

from ..core import STARTING_FEN, new_board
from ..core.types import Move
from ..protocol.engine import Analysis, Engine, EngineInfo, ScoredMove, SearchLimits
from ..tokenizer.board_tokenizer import BoardTokenizer
from ..tokenizer.move_tokenizer import MoveTokenizer

_INSTALL_HINT = "The board engine needs the 'model' extra: uv sync --extra model"

# A small fallback config for random-init inference (no checkpoint needed).
_FALLBACK_PRESET = "m1-dev"
_FALLBACK_N_LAYER = 2
_FALLBACK_N_EMBD = 64
_FALLBACK_N_HEAD = 2


def _local_resolve_device(requested: str) -> str:
    """Minimal device resolver — prefer WP3's version but fall back locally."""
    try:
        from ..training.board_loop import resolve_device_dtype  # noqa: PLC0415

        device, _ = resolve_device_dtype()
        # If caller asked for a specific non-auto device, honour it.
        if requested not in ("auto", ""):
            return requested
        return device
    except (ImportError, AttributeError):
        pass
    # Local fallback: trust whatever the caller passed; default cpu.
    return requested if requested not in ("auto", "") else "cpu"


class BoardTransformerEngine(Engine):
    """Board-state transformer :class:`Engine` with legal-masked policy decoding.

    Args:
        checkpoint: Path to a checkpoint saved by :meth:`BoardTransformer.save`.
            If ``None``, a tiny random-init model is used (always legal moves).
        device: PyTorch device string (``"cpu"``, ``"cuda"``, ``"mps"``, …).
    """

    def __init__(self, checkpoint: str | None = None, device: str = "cpu") -> None:
        try:
            import torch  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(_INSTALL_HINT) from exc
        self._torch = torch
        self._device = _local_resolve_device(device)
        self._board_tok = BoardTokenizer()
        self._move_tok = MoveTokenizer()
        self._temperature = 0.0  # 0 = argmax
        self._top_k = 0  # 0 = no top-k restriction
        self._rng = torch.Generator().manual_seed(0)
        self._checkpoint: str | None = checkpoint

        self._model, self._extra = self._load_model(checkpoint)

        self._board = new_board(STARTING_FEN)

    # -- Model loading -------------------------------------------------------

    def _load_model(self, checkpoint: str | None) -> tuple[Any, dict[str, Any]]:
        """Load or build the BoardTransformer; return ``(model, extra)``."""
        from ..model.board_transformer import (  # noqa: PLC0415
            BoardTransformer,
            BoardTransformerConfig,
        )

        if checkpoint is not None:
            model, extra = BoardTransformer.load(checkpoint, map_location=self._device)
        else:
            # Tiny random-init fallback — fast for conformance / tests.
            cfg = BoardTransformerConfig(
                d_model=_FALLBACK_N_EMBD,
                n_layer=_FALLBACK_N_LAYER,
                n_head=_FALLBACK_N_HEAD,
                ffn_hidden=_FALLBACK_N_EMBD * 2,
            )
            model, extra = BoardTransformer(cfg), {}

        model.to(self._device).eval()
        # Disable gradient computation for inference.
        for p in model.parameters():
            p.requires_grad_(False)

        return model, extra

    # -- Engine protocol -----------------------------------------------------

    def id(self) -> EngineInfo:
        step = self._extra.get("step")
        name = "board-transformer" + (f" (step {step})" if step is not None else "")
        return EngineInfo(
            name=name,
            author="Dong Feng contributors",
            options={"Temperature": "0.0", "TopK": "0", "Seed": "0"},
        )

    def new_game(self) -> None:
        self._board = new_board(STARTING_FEN)

    def set_position(self, fen: str, moves: list[Move]) -> None:
        self._board = new_board(fen)
        for m in moves:
            self._board.push(m)

    # -- Core inference helpers ----------------------------------------------

    def _forward(self) -> tuple[Any, float]:
        """Encode the current board and run a forward pass.

        Returns:
            ``(policy_logits_1d, value_scalar)`` — logits over 2554 moves and
            the scalar value in (−1, 1) from the side-to-move's perspective.
        """
        torch = self._torch
        ids = self._board_tok.encode(self._board.fen())
        boards = torch.tensor([ids], dtype=torch.long, device=self._device)  # [1, 91]
        with torch.no_grad():
            policy_logits, value_t = self._model(boards)
        # policy_logits: [1, 2554] → [2554]; value_t: [1] → scalar float
        return policy_logits[0], float(value_t[0].item())

    def _masked_scores(self) -> tuple[list[Move], Any, float]:
        """Return ``(legal_moves, probabilities_over_legal, value_scalar)``.

        ``probabilities`` is a 1-D tensor aligned with ``legal_moves`` (or
        ``None`` when there are no legal moves).
        """
        torch = self._torch
        legal = self._board.legal_moves()
        policy_logits, value = self._forward()

        if not legal:
            return [], None, value

        # Build legal id tensor and index into policy logits.
        legal_ids = torch.tensor(
            [self._move_tok.encode_move(m) for m in legal],
            dtype=torch.long,
            device=self._device,
        )  # [L]
        legal_logits = policy_logits[legal_ids]  # [L]

        # Temperature scaling — use temp=1.0 for softmax when argmax mode (temp≤0).
        temp = self._temperature if self._temperature > 0 else 1.0
        probs = torch.softmax(legal_logits / temp, dim=-1)  # [L]

        return legal, probs, value

    # -- bestmove / analyze --------------------------------------------------

    def bestmove(self, limits: SearchLimits) -> Move:
        legal, probs, _ = self._masked_scores()
        if not legal:
            raise ValueError("no legal moves in the current position (game over)")
        torch = self._torch

        if self._temperature <= 0:
            # Argmax over legal moves.
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
        legal, probs, value = self._masked_scores()
        if not legal:
            return Analysis(moves=[])

        # win_prob: map tanh scalar (−1, 1) → [0, 1] for side-to-move.
        win_prob = (value + 1.0) / 2.0

        # Sort by policy probability descending.
        order = sorted(range(len(legal)), key=lambda i: float(probs[i]), reverse=True)
        scored = [
            ScoredMove(
                move=legal[i],
                policy_prob=float(probs[i]),
                win_prob=win_prob,
            )
            for i in order
        ]
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
            self._model, self._extra = self._load_model(value)
            self._checkpoint = value

    def stop(self) -> None:
        """No-op: single-forward-pass inference has nothing to interrupt."""
