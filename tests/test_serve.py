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


def test_undo_takes_back_move_pair() -> None:
    session = _session()
    session.human_move("h2", "e2")  # human + engine reply -> 2 plies
    state = session.undo()
    assert state["history"] == []  # both plies removed
    assert state["turn"] == "red"  # human's turn again


def test_undo_with_no_moves_is_noop() -> None:
    session = _session()
    state = session.undo()
    assert state["history"] == []
    assert state["turn"] == "red"


def test_undo_when_human_is_black_keeps_engine_opening() -> None:
    session = _session()
    session.reset("random", "black", 0.0)  # engine opens -> 1 ply
    session.human_move("h9", "g7")  # human + engine reply -> 3 plies
    state = session.undo()
    assert len(state["history"]) == 1  # back to just the engine's opening
    assert state["turn"] == "black"  # human's turn again
