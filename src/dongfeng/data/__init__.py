"""Dong Feng data layer: raw game records -> tokenized training shards.

Public surface:
    GameSource, Game, Sample        -- data contracts (Protocol + dataclasses)
    parse_pgn / parse_xqf / ...     -- per-format ingestion parsers (M1)
    parse_file / iter_games_in      -- extension-dispatching ingestion (M1)
    build_shards / iter_samples     -- dataset building (M1)
    BuildStats                      -- summary of a shard build

The parsers delegate to the ``cchess`` backend; the dataset builder emits
autoregressive ``uint16`` token shards plus a ``dataset_meta.json``.
"""

from __future__ import annotations

from .base import Game, GameSource, Sample
from .dataset import BuildStats, build_shards, iter_samples
from .ingest import (
    iter_games_in,
    parse_cbf,
    parse_cbl,
    parse_cbr,
    parse_dhtmlxq,
    parse_dhtmlxq_file,
    parse_file,
    parse_pgn,
    parse_txt,
    parse_xqf,
)

__all__ = [
    "BuildStats",
    "Game",
    "GameSource",
    "Sample",
    "build_shards",
    "iter_games_in",
    "iter_samples",
    "parse_cbf",
    "parse_cbl",
    "parse_cbr",
    "parse_dhtmlxq",
    "parse_dhtmlxq_file",
    "parse_file",
    "parse_pgn",
    "parse_txt",
    "parse_xqf",
]
