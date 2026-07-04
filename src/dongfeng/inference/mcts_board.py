"""Tree-of-Thought MCTS engine around the board-state transformer (WP-MCTS).

:class:`MctsBoardEngine` wraps :class:`~dongfeng.model.board_transformer.BoardTransformer`
in an AlphaZero-style **PUCT** search, turning the model's single forward pass
(policy + value) into a tree search. It reuses the model loading and the
legal-move ``move-v1`` id -> logit masking from
:class:`~dongfeng.inference.board_engine.BoardTransformerEngine`.

Algorithm (PUCT)
----------------
* **Node** stores ``N`` (visits), ``W`` (value sum), ``Q = W/N``, ``P`` (prior),
  ``children: dict[Move, _Node]`` and a terminal flag. The board state is walked
  via ``core.board`` push/pop along the descent path (the root board is cloned
  once per :meth:`analyze`; a fresh clone is descended per simulation).
* **Select** — descend by
  ``argmax_a  Q(a) + c_puct * P(a) * sqrt(sum_b N(b)) / (1 + N(a))``.
* **Expand** — at an unexpanded leaf, one model forward -> policy logits masked
  to legal moves -> softmax priors ``P``; a leaf value ``v`` per ``value_mode``.
* **Backup** — propagate ``v`` up the path, **negating each ply** (zero-sum,
  two-player game).
* **Terminal** — ``board.is_game_over()`` / ``result()``. In **Xiangqi, no legal
  moves is a LOSS** for the side to move, so a terminal leaf has value ``-1``
  from that node's perspective (never a draw).

Not handled (deferred to M5)
----------------------------
Repetition and perpetual-check / perpetual-chase rules are **not** modelled: the
search treats positions purely by their static board state and never adjudicates
a repeated line as a draw or a loss.

Value modes (``value_mode`` option)
-----------------------------------
* ``"head"`` — use the model's value head (best once M4 distillation labels
  exist).
* ``"rollout"`` — play to a terminal (or a depth cap) with the policy for a leaf
  estimate; works **today** without a trained value head. **Default.**
* ``"zero"`` — priors + terminals only (pure policy-guided tree).

**Honest gating:** the current value head was trained on an all-masked corpus
(no result labels), so ``"head"`` values are close to noise. ``"rollout"`` is the
default for that reason; real strength needs M4 distillation labels.

The ``torch`` import is deferred to construction so importing this module never
requires the optional ``model`` extra.
"""

from __future__ import annotations

import math
import time
from typing import Any

from ..core import STARTING_FEN, new_board
from ..core.types import GameResult, Move
from ..protocol.engine import Analysis, Engine, EngineInfo, ScoredMove, SearchLimits
from ..tokenizer.board_tokenizer import BoardTokenizer
from ..tokenizer.move_tokenizer import MoveTokenizer

_INSTALL_HINT = "The board MCTS engine needs the 'model' extra: uv sync --extra model"

# A small fallback config for random-init inference (no checkpoint needed).
_FALLBACK_N_LAYER = 2
_FALLBACK_N_EMBD = 64
_FALLBACK_N_HEAD = 2

# Default search knobs.
_DEFAULT_C_PUCT = 2.0
_DEFAULT_N_SIMULATIONS = 200
_ROLLOUT_MAX_PLIES = 60  # safety cap for the rollout value estimate

# Transposition (evaluation) cache: positions recur heavily across simulations
# and rollouts, and the model is deterministic per checkpoint, so (legal,
# priors, value) can be memoised by FEN. Cleared wholesale when full — an LRU
# would cost more bookkeeping than the forwards it saves at this size.
_EVAL_CACHE_MAX = 100_000


def _local_resolve_device(requested: str) -> str:
    """Minimal device resolver — prefer WP3's version but fall back locally."""
    try:
        from ..training.board_loop import (  # noqa: PLC0415  # pyright: ignore[reportMissingImports]
            resolve_device_dtype,
        )

        device, _ = resolve_device_dtype()
        if requested not in ("auto", ""):
            return requested
        return device
    except (ImportError, AttributeError):
        pass
    return requested if requested not in ("auto", "") else "cpu"


class _Node:
    """A single PUCT search node."""

    __slots__ = ("children", "expanded", "is_terminal", "n", "p", "w")

    def __init__(self, prior: float) -> None:
        self.n: int = 0
        self.w: float = 0.0
        self.p: float = prior
        self.children: dict[Move, _Node] = {}
        self.is_terminal: bool = False
        self.expanded: bool = False

    @property
    def q(self) -> float:
        return self.w / self.n if self.n > 0 else 0.0


class MctsBoardEngine(Engine):
    """PUCT (AlphaZero-style) MCTS :class:`Engine` around the board transformer.

    Args:
        checkpoint: Path to a checkpoint saved by :meth:`BoardTransformer.save`.
            If ``None``, a tiny random-init model is used (always legal moves).
        device: PyTorch device string (``"cpu"``, ``"cuda"``, ``"mps"``, ``"auto"``).
        c_puct: Exploration constant in the PUCT selection score.
        n_simulations: Default number of simulations per search (overridable via
            ``SearchLimits.nodes``).
        value_mode: One of ``"head"``, ``"rollout"`` (default), ``"zero"``.
        temperature: Move-selection temperature. ``<= 0`` picks the max-visit
            child deterministically; ``> 0`` samples proportional to visit counts.
        dirichlet: Root Dirichlet-noise weight in ``[0, 1]`` (self-play
            exploration); ``0`` disables it.
        seed: RNG seed for sampling / Dirichlet noise.
    """

    def __init__(
        self,
        checkpoint: str | None = None,
        device: str = "cpu",
        *,
        c_puct: float = _DEFAULT_C_PUCT,
        n_simulations: int = _DEFAULT_N_SIMULATIONS,
        value_mode: str = "rollout",
        temperature: float = 0.0,
        dirichlet: float = 0.0,
        seed: int = 0,
    ) -> None:
        try:
            import torch  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(_INSTALL_HINT) from exc
        self._torch = torch
        self._device = _local_resolve_device(device)
        self._board_tok = BoardTokenizer()
        self._move_tok = MoveTokenizer()

        self._c_puct = float(c_puct)
        self._n_simulations = int(n_simulations)
        self._value_mode = value_mode
        self._temperature = float(temperature)
        self._dirichlet = float(dirichlet)
        self._seed = int(seed)
        self._rng = torch.Generator().manual_seed(self._seed)

        self._stop_flag = False
        self._checkpoint: str | None = checkpoint
        self._model, self._extra = self._load_model(checkpoint)

        # bf16 autocast on CUDA halves inference bandwidth/compute; mps/cpu stay
        # fp32 (fp16 on MPS risks NaN and the win is small at batch 1).
        self._amp_dtype = torch.bfloat16 if self._device.startswith("cuda") else None

        # FEN -> (legal_moves, priors, head_value); valid per checkpoint.
        self._eval_cache: dict[str, tuple[list[Move], list[float], float]] = {}
        # Perf counters for the last _run_search (exposed via last_search_stats).
        self._nn_forwards = 0
        self._cache_hits = 0
        self.last_search_stats: dict[str, Any] = {}

        self._board = new_board(STARTING_FEN)

    # -- Model loading -------------------------------------------------------

    def _load_model(self, checkpoint: str | None) -> tuple[Any, dict[str, Any]]:
        from ..model.board_transformer import (  # noqa: PLC0415
            BoardTransformer,
            BoardTransformerConfig,
        )

        if checkpoint is not None:
            model, extra = BoardTransformer.load(checkpoint, map_location=self._device)
        else:
            cfg = BoardTransformerConfig(
                d_model=_FALLBACK_N_EMBD,
                n_layer=_FALLBACK_N_LAYER,
                n_head=_FALLBACK_N_HEAD,
                ffn_hidden=_FALLBACK_N_EMBD * 2,
            )
            model, extra = BoardTransformer(cfg), {}

        model.to(self._device).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        return model, extra

    # -- Engine protocol -----------------------------------------------------

    def id(self) -> EngineInfo:
        step = self._extra.get("step")
        name = "board-mcts" + (f" (step {step})" if step is not None else "")
        return EngineInfo(
            name=name,
            author="Dong Feng contributors",
            options={
                "c_puct": str(self._c_puct),
                "n_simulations": str(self._n_simulations),
                "value_mode": self._value_mode,
                "temperature": str(self._temperature),
                "dirichlet": str(self._dirichlet),
                "seed": str(self._seed),
            },
        )

    def new_game(self) -> None:
        self._board = new_board(STARTING_FEN)

    def set_position(self, fen: str, moves: list[Move]) -> None:
        self._board = new_board(fen)
        for m in moves:
            self._board.push(m)

    def set_option(self, name: str, value: str) -> None:
        key = name.lower()
        if key == "c_puct":
            self._c_puct = float(value)
        elif key == "n_simulations":
            self._n_simulations = int(value)
        elif key == "value_mode":
            if value not in ("head", "rollout", "zero"):
                raise ValueError(f"unknown value_mode {value!r}; choose head | rollout | zero")
            self._value_mode = value
        elif key == "temperature":
            self._temperature = float(value)
        elif key == "dirichlet":
            self._dirichlet = float(value)
        elif key == "seed":
            self._seed = int(value)
            self._rng = self._torch.Generator().manual_seed(self._seed)
        elif key == "checkpoint":
            self._model, self._extra = self._load_model(value)
            self._checkpoint = value
            self._eval_cache.clear()  # cached evals belong to the old weights

    def stop(self) -> None:
        """Cooperatively request the running simulation loop to halt."""
        self._stop_flag = True

    # -- Model / policy helpers ----------------------------------------------

    def _policy_value(self, board: Any) -> tuple[list[Move], list[float], float]:
        """Evaluate ``board``, memoised by FEN (transposition cache).

        Returns ``(legal_moves, priors, head_value)`` where ``priors`` is a list
        aligned with ``legal_moves`` (softmax over legal logits) and
        ``head_value`` is the value-head scalar in (-1, 1) from ``board``'s
        side-to-move perspective. ``legal_moves`` is empty for a terminal node.
        """
        fen = board.fen()
        cached = self._eval_cache.get(fen)
        if cached is not None:
            self._cache_hits += 1
            return cached

        torch = self._torch
        legal = board.legal_moves()
        ids = self._board_tok.encode(fen)
        boards = torch.tensor([ids], dtype=torch.long, device=self._device)
        self._nn_forwards += 1
        with torch.inference_mode():
            if self._amp_dtype is not None:
                with torch.autocast(device_type="cuda", dtype=self._amp_dtype):
                    policy_logits, value_t = self._model(boards)
            else:
                policy_logits, value_t = self._model(boards)
        head_value = float(value_t[0].item())
        if not legal:
            result: tuple[list[Move], list[float], float] = ([], [], head_value)
        else:
            legal_ids = torch.tensor(
                [self._move_tok.encode_move(m) for m in legal],
                dtype=torch.long,
                device=self._device,
            )
            legal_logits = policy_logits[0].float()[legal_ids]
            probs = torch.softmax(legal_logits, dim=-1)
            result = (legal, [float(x) for x in probs.tolist()], head_value)

        if len(self._eval_cache) >= _EVAL_CACHE_MAX:
            self._eval_cache.clear()
        self._eval_cache[fen] = result
        return result

    def _rollout_value(self, board: Any) -> float:
        """Play a policy-greedy rollout to a terminal (or cap); return value.

        The value is from the perspective of the side to move in ``board`` at
        entry: +1 win / -1 loss / 0 draw-by-cap.
        """
        sim = board.clone()
        plies = 0
        while plies < _ROLLOUT_MAX_PLIES:
            if sim.is_game_over():
                # Side to move in `sim` has no moves -> that side loses.
                loser_is_root = plies % 2 == 0
                return -1.0 if loser_is_root else 1.0
            legal, priors, _ = self._policy_value(sim)
            if not legal:  # pragma: no cover - covered by is_game_over above
                loser_is_root = plies % 2 == 0
                return -1.0 if loser_is_root else 1.0
            best_i = max(range(len(legal)), key=lambda i: priors[i])
            sim.push(legal[best_i])
            plies += 1
        return 0.0  # depth cap reached -> treat as a draw estimate

    def _leaf_value(self, board: Any, head_value: float, is_terminal: bool) -> float:
        """Compute the leaf value from ``board``'s side-to-move perspective."""
        if is_terminal:
            # No legal moves in Xiangqi == loss for the side to move.
            return -1.0
        if self._value_mode == "head":
            return head_value
        if self._value_mode == "rollout":
            return self._rollout_value(board)
        # "zero": priors + terminals only.
        return 0.0

    # -- PUCT search ---------------------------------------------------------

    def _select_child(self, node: _Node) -> Move:
        sqrt_total = math.sqrt(max(node.n, 1))
        best_move: Move | None = None
        best_score = -float("inf")
        for move, child in node.children.items():
            u = self._c_puct * child.p * sqrt_total / (1 + child.n)
            score = child.q + u
            if score > best_score:
                best_score = score
                best_move = move
        assert best_move is not None
        return best_move

    def _expand(self, node: _Node, board: Any) -> float:
        """Expand ``node`` for the position in ``board``; return its leaf value."""
        if board.is_game_over():
            node.is_terminal = True
            node.expanded = True
            return -1.0  # side to move lost
        legal, priors, head_value = self._policy_value(board)
        for move, prior in zip(legal, priors, strict=True):
            node.children[move] = _Node(prior)
        node.expanded = True
        return self._leaf_value(board, head_value, is_terminal=False)

    def _add_root_noise(self, root: _Node) -> None:
        if self._dirichlet <= 0.0 or not root.children:
            return
        # Dirichlet(alpha) sampling via normalized Gamma draws, using the engine's
        # torch Generator so noise is reproducible from the seed.
        torch = self._torch
        k = len(root.children)
        conc = torch.full((k,), 0.3)
        gammas = torch._standard_gamma(conc, generator=self._rng)  # type: ignore[attr-defined]
        noise = (gammas / gammas.sum()).tolist()
        eps = self._dirichlet
        for (_move, child), nz in zip(root.children.items(), noise, strict=True):
            child.p = (1 - eps) * child.p + eps * nz

    def _run_search(self, limits: SearchLimits) -> tuple[_Node, int]:
        """Build a PUCT tree from the current position. Returns ``(root, sims)``."""
        self._stop_flag = False
        self._nn_forwards = 0
        self._cache_hits = 0
        t0 = time.monotonic()
        root_board = self._board.clone()
        root = _Node(prior=1.0)
        # Prime the root so it has children (and priors) before selection.
        self._expand(root, root_board)
        self._add_root_noise(root)

        n_sims = limits.nodes if limits.nodes is not None else self._n_simulations
        deadline = None
        if limits.movetime_ms is not None:
            deadline = time.monotonic() + limits.movetime_ms / 1000.0

        sims = 0
        if not (root.is_terminal or not root.children):
            while sims < n_sims:
                if self._stop_flag:
                    break
                if deadline is not None and time.monotonic() >= deadline:
                    break
                self._simulate(root, root_board)
                sims += 1

        # Search-perf metrics for the last search (sims/s, NN forwards saved by
        # the transposition cache). Not part of the Engine protocol — read via
        # engine.last_search_stats by callers that want perf telemetry.
        dt = max(time.monotonic() - t0, 1e-9)
        lookups = self._nn_forwards + self._cache_hits
        self.last_search_stats = {
            "sims": sims,
            "time_ms": int(dt * 1000),
            "sims_per_s": round(sims / dt, 1),
            "nn_forwards": self._nn_forwards,
            "cache_hits": self._cache_hits,
            "cache_hit_rate": round(self._cache_hits / lookups, 4) if lookups else 0.0,
            "cache_size": len(self._eval_cache),
        }
        return root, sims

    def _simulate(self, root: _Node, root_board: Any) -> None:
        """Run one selection->expansion->backup simulation from the root."""
        board = root_board.clone()
        path: list[_Node] = [root]
        node = root
        # Descend while the node is expanded, non-terminal, and has children.
        while node.expanded and not node.is_terminal and node.children:
            move = self._select_child(node)
            board.push(move)
            node = node.children[move]
            path.append(node)

        # Terminal (no moves -> loss) or an expanded childless node (rare) -> a
        # terminal loss for the side to move here; otherwise expand this leaf.
        already_leaf = node.is_terminal or node.expanded
        value = -1.0 if already_leaf else self._expand(node, board)

        # Backup, negating each ply (zero-sum). `value` is from the perspective
        # of the side to move at the leaf.
        for ancestor in reversed(path):
            ancestor.n += 1
            ancestor.w += value
            value = -value

    # -- bestmove / analyze --------------------------------------------------

    def _scored_moves(self, root: _Node) -> list[ScoredMove]:
        # win_prob from root Q (root Q is the mover's expected value in (-1, 1)).
        root_win_prob = (root.q + 1.0) / 2.0
        items = sorted(root.children.items(), key=lambda kv: kv[1].n, reverse=True)
        total = sum(child.n for _, child in items) or 1
        return [
            ScoredMove(
                move=move,
                policy_prob=child.n / total,
                win_prob=root_win_prob,
            )
            for move, child in items
        ]

    def bestmove(self, limits: SearchLimits) -> Move:
        root, _ = self._run_search(limits)
        if not root.children:
            raise ValueError("no legal moves in the current position (game over)")
        torch = self._torch
        moves = list(root.children.keys())
        visits = torch.tensor([float(root.children[m].n) for m in moves])

        if self._temperature <= 0:
            idx = int(torch.argmax(visits).item())
        else:
            weights = visits ** (1.0 / self._temperature)
            total = float(weights.sum().item())
            if total <= 0:
                idx = int(torch.argmax(visits).item())
            else:
                idx = int(torch.multinomial(weights / total, 1, generator=self._rng).item())
        return moves[idx]

    def analyze(self, limits: SearchLimits) -> Analysis:
        start = time.monotonic()
        root, sims = self._run_search(limits)
        if not root.children:
            return Analysis(moves=[])
        elapsed_ms = int((time.monotonic() - start) * 1000)
        return Analysis(
            moves=self._scored_moves(root),
            nodes=sims,
            time_ms=elapsed_ms,
        )

    # -- helpers for result reporting ---------------------------------------

    def result(self) -> GameResult:
        """Current board's game result (helper; not part of the Engine Protocol)."""
        return self._board.result()
