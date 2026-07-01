"""Tests for the web-play game session (M3). Uses the random engine (no torch)."""

from __future__ import annotations

from dongfeng.serve.webplay import GameSession


def _session() -> GameSession:
    return GameSession("random", None)


def test_initial_state_shape() -> None:
    s = _session().state()
    assert s["turn"] == "red"
    assert s["human"] == "red"
    assert s["result"] == "ongoing"
    assert len(s["legal"]) == 44  # legal moves from the start position
    assert all(len(m) == 2 for m in s["legal"])


def test_human_move_then_engine_reply() -> None:
    session = _session()
    res = session.human_move("h2", "e2")  # cannon to centre
    assert res["error"] is None
    assert res["engine_move"] is not None  # engine (Black) replied
    state = res["state"]
    assert len(state["history"]) == 2  # human + engine
    assert state["turn"] == "red"  # back to the human


def test_illegal_move_rejected() -> None:
    session = _session()
    res = session.human_move("h2", "h2")
    assert res["error"] is not None
    assert res["state"]["history"] == []


def test_reset_with_human_black_lets_engine_open() -> None:
    session = _session()
    state = session.reset("random", "black", 0.0)
    assert state["human"] == "black"
    assert len(state["history"]) == 1  # engine (Red) has already opened
    assert state["turn"] == "black"
