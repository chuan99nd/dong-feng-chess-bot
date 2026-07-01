"""Tests for the BC pretrain loop (M2). Skipped without torch."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("torch")

from dongfeng.core import STARTING_FEN, GameResult, Move  # noqa: E402
from dongfeng.data import Game, build_shards  # noqa: E402
from dongfeng.model import TransformerConfig, TransformerPolicy  # noqa: E402
from dongfeng.tokenizer import MoveTokenizer  # noqa: E402
from dongfeng.training.base import TrainConfig  # noqa: E402
from dongfeng.training.loop import bc_pretrain, load_token_stream  # noqa: E402

_MOVES = [Move.from_iccs(s) for s in ("h2e2", "h9g7", "h0g2", "i9h9")]


def test_bc_pretrain_writes_loadable_checkpoint(tmp_path: Path) -> None:
    data_dir = tmp_path / "shards"
    games = [Game(start_fen=STARTING_FEN, moves=list(_MOVES), result=GameResult.RED_WIN)] * 50
    build_shards(games, data_dir)

    model = TransformerPolicy(
        TransformerConfig(vocab_size=MoveTokenizer().vocab_size, n_layer=1, n_embd=32, n_head=2,
                          block_size=16)
    )
    cfg = TrainConfig(
        data_dir=data_dir,
        out_dir=tmp_path / "out",
        batch_size=4,
        max_steps=3,
        warmup_steps=1,
        checkpoint_every=1,
        device="cpu",
    )
    ckpt = bc_pretrain(model, cfg)
    assert ckpt.exists()
    loaded, extra = TransformerPolicy.load(ckpt)
    assert extra["tokenizer"] == "move-v1"


def test_load_token_stream_roundtrips_counts(tmp_path: Path) -> None:
    games = [Game(start_fen=STARTING_FEN, moves=list(_MOVES), result=GameResult.DRAW)]
    stats = build_shards(games, tmp_path)
    stream = load_token_stream(tmp_path)
    assert len(stream) == stats.num_tokens
