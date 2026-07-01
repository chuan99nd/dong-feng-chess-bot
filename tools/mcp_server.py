#!/usr/bin/env python3
"""Dong Feng MCP server — token-efficient queries over ``manifest.json``.

This exposes small, structured query tools so that agents can inspect datasets,
checkpoints, runs, positions, and the tokenizer **without reading large artifacts**
(training shards, checkpoint weights, run logs — all git-ignored under ``data/``,
``checkpoints/``, ``runs/``). Every tool returns a few hundred tokens of JSON at
most; the source of truth is ``manifest.json`` at the repo root.

Tools:
    - ``dataset_stats``        summary of indexed datasets (counts, sources)
    - ``checkpoint_info(id?)`` metadata for a checkpoint (or all of them)
    - ``eval_last``            the most recent eval run's headline metrics
    - ``position_query(fen)``  legal-move summary for a FEN (uses the rules backend)
    - ``tokenizer_info(id?)``  tokenizer vocab summary

Transport is **stdio**. If the optional ``mcp`` Python SDK is installed, a real
FastMCP server is started; otherwise this falls back to a tiny hand-rolled
stdio JSON-RPC 2.0 stub implementing just enough of MCP (``initialize``,
``tools/list``, ``tools/call``) to be usable. The stub is CLEARLY MARKED and is a
best-effort convenience, not a full MCP implementation.

Run:
    uv run --extra mcp python tools/mcp_server.py     # real SDK if available
    python tools/mcp_server.py                        # stub fallback otherwise
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

# --------------------------------------------------------------------------- #
# manifest access (shared by both the SDK server and the stub)
# --------------------------------------------------------------------------- #

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MANIFEST_PATH = os.environ.get("DONGFENG_MANIFEST", os.path.join(_REPO_ROOT, "manifest.json"))


def _load_manifest() -> dict[str, Any]:
    """Load and parse ``manifest.json``; return ``{}`` on any problem."""
    try:
        with open(_MANIFEST_PATH, encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def dataset_stats() -> dict[str, Any]:
    """Summarize indexed datasets: totals plus a compact per-dataset list."""
    m = _load_manifest()
    datasets = m.get("datasets") or []
    total_games = sum(int(d.get("num_games") or 0) for d in datasets)
    total_samples = sum(int(d.get("num_samples") or 0) for d in datasets)
    return {
        "count": len(datasets),
        "total_games": total_games,
        "total_samples": total_samples,
        "datasets": [
            {
                "id": d.get("id"),
                "source": d.get("source"),
                "format": d.get("format"),
                "num_games": d.get("num_games"),
                "num_samples": d.get("num_samples"),
                "tokenizer": d.get("tokenizer"),
            }
            for d in datasets
        ],
        "note": "empty until M1 (data + tokenizer) lands",
    }


def checkpoint_info(checkpoint_id: str | None = None) -> dict[str, Any]:
    """Return metadata for one checkpoint (by id) or all indexed checkpoints."""
    m = _load_manifest()
    checkpoints = m.get("checkpoints") or []
    if checkpoint_id is not None:
        match = next((c for c in checkpoints if c.get("id") == checkpoint_id), None)
        if match is None:
            return {"error": f"no checkpoint with id {checkpoint_id!r}", "known": len(checkpoints)}
        return match
    return {
        "count": len(checkpoints),
        "checkpoints": [
            {
                "id": c.get("id"),
                "arch": c.get("arch"),
                "step": c.get("step"),
                "metrics": c.get("metrics"),
            }
            for c in checkpoints
        ],
        "note": "empty until M2 (training) lands",
    }


def eval_last() -> dict[str, Any]:
    """Return the most recently finished eval run's headline metrics."""
    m = _load_manifest()
    runs = [r for r in (m.get("runs") or []) if r.get("kind") == "eval"]
    if not runs:
        return {"error": "no eval runs indexed yet", "note": "populated from M2"}

    def _key(r: dict[str, Any]) -> str:
        return str(r.get("finished") or r.get("started") or "")

    last = max(runs, key=_key)
    return {
        "id": last.get("id"),
        "status": last.get("status"),
        "checkpoint": last.get("checkpoint"),
        "metrics": last.get("metrics"),
        "finished": last.get("finished"),
    }


def position_query(fen: str) -> dict[str, Any]:
    """Summarize a FEN: side to move, legal-move count/list, and status.

    Uses the rules backend (``dongfeng.core``). If the backend or its dependency
    (``cchess``) is unavailable, returns an ``error`` field instead of raising.
    """
    try:
        from dongfeng.core import GameResult, new_board  # noqa: PLC0415
        from dongfeng.core.fen import side_to_move  # noqa: PLC0415
    except Exception as exc:  # pragma: no cover - environment dependent
        return {"error": f"rules backend unavailable: {type(exc).__name__}: {exc}", "fen": fen}

    try:
        board = new_board(fen)
    except Exception as exc:
        return {"error": f"invalid FEN or backend error: {type(exc).__name__}: {exc}", "fen": fen}

    legal = [m.iccs for m in board.legal_moves()]
    result = board.result()
    return {
        "fen": fen,
        "side_to_move": side_to_move(fen),
        "num_legal_moves": len(legal),
        "legal_moves": legal,
        "is_check": board.is_check(),
        "is_game_over": board.is_game_over(),
        "result": result.value if isinstance(result, GameResult) else str(result),
    }


def tokenizer_info(tokenizer_id: str | None = None) -> dict[str, Any]:
    """Return a tokenizer's vocab summary (by id) or all indexed tokenizers."""
    m = _load_manifest()
    tokenizers = m.get("tokenizers") or []
    if tokenizer_id is not None:
        match = next((t for t in tokenizers if t.get("id") == tokenizer_id), None)
        if match is None:
            return {"error": f"no tokenizer with id {tokenizer_id!r}", "known": len(tokenizers)}
        return match
    return {
        "count": len(tokenizers),
        "tokenizers": [{"id": t.get("id"), "vocab_size": t.get("vocab_size")} for t in tokenizers],
        "note": "empty until M1 (tokenizer) lands",
    }


# Tool registry shared by the SDK server and the stub: name -> (fn, schema).
_TOOLS: dict[str, dict[str, Any]] = {
    "dataset_stats": {
        "fn": dataset_stats,
        "description": "Summary of indexed datasets (counts, sources). Reads manifest.json.",
        "schema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    "checkpoint_info": {
        "fn": checkpoint_info,
        "description": "Checkpoint metadata by id, or all checkpoints if id omitted.",
        "schema": {
            "type": "object",
            "properties": {"checkpoint_id": {"type": "string"}},
            "additionalProperties": False,
        },
    },
    "eval_last": {
        "fn": eval_last,
        "description": "The most recent eval run's headline metrics.",
        "schema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    "position_query": {
        "fn": position_query,
        "description": "Legal-move summary for a Xiangqi FEN (uses the rules backend).",
        "schema": {
            "type": "object",
            "properties": {"fen": {"type": "string"}},
            "required": ["fen"],
            "additionalProperties": False,
        },
    },
    "tokenizer_info": {
        "fn": tokenizer_info,
        "description": "Tokenizer vocab summary by id, or all tokenizers if id omitted.",
        "schema": {
            "type": "object",
            "properties": {"tokenizer_id": {"type": "string"}},
            "additionalProperties": False,
        },
    },
}


# --------------------------------------------------------------------------- #
# Preferred path: the official `mcp` SDK (FastMCP), if installed.
# --------------------------------------------------------------------------- #


def _run_with_sdk() -> bool:
    """Try to run a real MCP server via the `mcp` SDK. Return False if unavailable."""
    try:
        from mcp.server.fastmcp import FastMCP  # noqa: PLC0415
    except Exception:
        return False

    server = FastMCP("dongfeng")

    # Register each tool with the SDK. Signatures are explicit so the SDK can
    # generate correct input schemas.
    @server.tool()
    def dataset_stats_tool() -> dict[str, Any]:
        """Summary of indexed datasets (counts, sources)."""
        return dataset_stats()

    @server.tool()
    def checkpoint_info_tool(checkpoint_id: str | None = None) -> dict[str, Any]:
        """Checkpoint metadata by id, or all checkpoints if id omitted."""
        return checkpoint_info(checkpoint_id)

    @server.tool()
    def eval_last_tool() -> dict[str, Any]:
        """The most recent eval run's headline metrics."""
        return eval_last()

    @server.tool()
    def position_query_tool(fen: str) -> dict[str, Any]:
        """Legal-move summary for a Xiangqi FEN (uses the rules backend)."""
        return position_query(fen)

    @server.tool()
    def tokenizer_info_tool(tokenizer_id: str | None = None) -> dict[str, Any]:
        """Tokenizer vocab summary by id, or all tokenizers if id omitted."""
        return tokenizer_info(tokenizer_id)

    server.run()  # stdio transport by default
    return True


# --------------------------------------------------------------------------- #
# Fallback: a tiny stdio JSON-RPC 2.0 MCP stub.  *** STUB — best effort. ***
# Implements just enough of MCP (initialize / tools/list / tools/call) to be
# usable when the `mcp` SDK is not installed. Not a complete implementation.
# --------------------------------------------------------------------------- #

_PROTOCOL_VERSION = "2024-11-05"


def _stub_result(request_id: Any, result: Any) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def _stub_error(request_id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


def _stub_handle(msg: dict[str, Any]) -> dict[str, Any] | None:
    """Handle one JSON-RPC message; return a response dict, or None for notifications."""
    method = msg.get("method")
    request_id = msg.get("id")
    params = msg.get("params") or {}

    if method == "initialize":
        return _stub_result(
            request_id,
            {
                "protocolVersion": _PROTOCOL_VERSION,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "dongfeng", "version": "0.0.0-stub"},
            },
        )
    if method == "notifications/initialized":
        return None  # notification, no response
    if method == "tools/list":
        return _stub_result(
            request_id,
            {
                "tools": [
                    {
                        "name": name,
                        "description": spec["description"],
                        "inputSchema": spec["schema"],
                    }
                    for name, spec in _TOOLS.items()
                ]
            },
        )
    if method == "tools/call":
        name = params.get("name")
        args = params.get("arguments") or {}
        spec = _TOOLS.get(name)
        if spec is None:
            return _stub_error(request_id, -32602, f"unknown tool: {name!r}")
        try:
            value = spec["fn"](**args)
        except TypeError as exc:
            return _stub_error(request_id, -32602, f"bad arguments: {exc}")
        except Exception as exc:  # noqa: BLE001 - report tool failure as content
            value = {"error": f"{type(exc).__name__}: {exc}"}
        return _stub_result(
            request_id,
            {"content": [{"type": "text", "text": json.dumps(value)}]},
        )
    if request_id is not None:
        return _stub_error(request_id, -32601, f"method not found: {method!r}")
    return None


def _run_stub() -> None:
    """Line-delimited JSON-RPC over stdio. *** STUB fallback. ***"""
    sys.stderr.write(
        "dongfeng MCP: 'mcp' SDK not found; running minimal stdio JSON-RPC STUB "
        "(install the optional 'mcp' extra for the full server).\n"
    )
    sys.stderr.flush()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        response = _stub_handle(msg)
        if response is not None:
            sys.stdout.write(json.dumps(response) + "\n")
            sys.stdout.flush()


def main() -> None:
    if not _run_with_sdk():
        _run_stub()


if __name__ == "__main__":
    main()
