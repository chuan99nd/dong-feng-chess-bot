"""Local web UI to play a human against a Dong Feng engine (stdlib only).

A single :class:`GameSession` holds the board, move history, and the chosen engine.
The browser renders the board (SVG) and posts moves to a tiny JSON API:

* ``GET  /``            -> the board page (HTML/CSS/JS, embedded below)
* ``GET  /api/state``   -> current game state (fen, legal moves, turn, result)
* ``POST /api/new``     -> reset the game (engine, human color, temperature)
* ``POST /api/move``    -> apply a human move, then the engine's reply

The engine runs server-side, so the neural :class:`~dongfeng.inference.transformer_engine.TransformerEngine`
(PyTorch) works exactly like the random baseline behind the same
:class:`~dongfeng.protocol.engine.Engine` contract.
"""

from __future__ import annotations

import json
import os
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from ..core import STARTING_FEN, new_board
from ..core.types import Color, Move
from ..protocol.engine import Engine, SearchLimits


def _make_engine(name: str, checkpoint: str | None) -> Engine:
    """Build an engine by name (``random`` / ``neural``) for the session."""
    if name == "neural":
        from ..inference.transformer_engine import TransformerEngine  # noqa: PLC0415

        return TransformerEngine(checkpoint or os.environ.get("DONGFENG_CKPT") or None)
    from ..engines import RandomEngine  # noqa: PLC0415

    return RandomEngine()


class GameSession:
    """A single in-memory game: board + history + engine (human vs engine)."""

    def __init__(self, engine_name: str, checkpoint: str | None) -> None:
        self._checkpoint = checkpoint
        self._lock = threading.Lock()
        self.reset(engine_name, "red", 0.0)

    def reset(self, engine_name: str, human: str, temperature: float) -> dict[str, Any]:
        with self._lock:
            self.board = new_board(STARTING_FEN)
            self.history: list[Move] = []
            self.engine_name = engine_name if engine_name in ("random", "neural") else "random"
            self.human = Color.RED if human == "red" else Color.BLACK
            self.temperature = temperature
            self.engine = _make_engine(self.engine_name, self._checkpoint)
            self.engine.new_game()
            self._configure_engine()
            self.last_move: list[str] | None = None
            # If the human plays Black, the engine (Red) opens.
            if self.human is Color.BLACK:
                self._engine_reply()
            return self._state()

    def _configure_engine(self) -> None:
        if self.engine_name == "neural":
            self.engine.set_option("Temperature", str(self.temperature))

    def _engine_reply(self) -> str | None:
        if self.board.is_game_over():
            return None
        self.engine.set_position(STARTING_FEN, list(self.history))
        move = self.engine.bestmove(SearchLimits(movetime_ms=200))
        self.board.push(move)
        self.history.append(move)
        self.last_move = [move.from_sq, move.to_sq]
        return move.iccs

    def human_move(self, frm: str, to: str) -> dict[str, Any]:
        with self._lock:
            if self.board.turn is not self.human:
                return {"error": "not your turn", "state": self._state()}
            try:
                move = Move(frm, to)
            except ValueError as exc:
                return {"error": str(exc), "state": self._state()}
            if not self.board.is_legal(move):
                return {"error": "illegal move", "state": self._state()}
            self.board.push(move)
            self.history.append(move)
            self.last_move = [frm, to]
            engine_move = self._engine_reply()
            return {"error": None, "engine_move": engine_move, "state": self._state()}

    def _state(self) -> dict[str, Any]:
        return {
            "fen": self.board.fen(),
            "turn": self.board.turn.value,
            "legal": [[m.from_sq, m.to_sq] for m in self.board.legal_moves()],
            "history": [m.iccs for m in self.history],
            "result": self.board.result().value,
            "in_check": self.board.is_check(),
            "human": self.human.value,
            "engine": self.engine_name,
            "temperature": self.temperature,
            "last_move": self.last_move,
        }

    def state(self) -> dict[str, Any]:
        with self._lock:
            return self._state()


def _make_handler(session: GameSession) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - matches base
            pass  # quiet: suppress default request logging

        def _send_json(self, obj: dict[str, Any], status: int = 200) -> None:
            body = json.dumps(obj).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", 0))
            if not length:
                return {}
            try:
                return json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError:
                return {}

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/" or self.path.startswith("/index"):
                body = _HTML.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/api/state":
                self._send_json(session.state())
            else:
                self.send_error(404)

        def do_POST(self) -> None:  # noqa: N802
            data = self._read_json()
            if self.path == "/api/new":
                self._send_json(
                    session.reset(
                        str(data.get("engine", "neural")),
                        str(data.get("human", "red")),
                        float(data.get("temperature", 0.0)),
                    )
                )
            elif self.path == "/api/move":
                self._send_json(session.human_move(str(data.get("from")), str(data.get("to"))))
            else:
                self.send_error(404)

    return Handler


def serve(
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
    engine: str = "neural",
    checkpoint: str | None = None,
    open_browser: bool = True,
) -> None:
    """Start the web-play server (blocking) and optionally open a browser."""
    session = GameSession(engine, checkpoint)
    httpd = ThreadingHTTPServer((host, port), _make_handler(session))
    url = f"http://{host}:{port}/"
    print(f"Dong Feng web UI on {url}  (engine: {engine})  — Ctrl+C to stop", flush=True)
    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping…", flush=True)
    finally:
        httpd.server_close()


# --------------------------------------------------------------------------- #
# Embedded single-page board UI (no external assets; strict-CSP friendly).
# --------------------------------------------------------------------------- #
_HTML = r"""<!doctype html>
<html lang="vi">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Dong Feng — Cờ tướng</title>
<style>
  :root { --wood:#e8c48c; --line:#5b3a1a; --red:#c0392b; --black:#1c1c1c; --cream:#f4e4c1; }
  * { box-sizing: border-box; }
  body { margin:0; font-family: system-ui, "PingFang SC", "Microsoft YaHei", sans-serif;
         background:#2b2b2b; color:#eee; display:flex; justify-content:center; padding:16px; }
  .wrap { display:flex; gap:20px; flex-wrap:wrap; align-items:flex-start; }
  h1 { font-size:20px; margin:0 0 10px; }
  #board { background:var(--wood); border-radius:6px; box-shadow:0 6px 24px rgba(0,0,0,.5); touch-action:manipulation; }
  .panel { min-width:230px; max-width:280px; }
  .card { background:#333; border-radius:8px; padding:12px 14px; margin-bottom:12px; }
  label { display:block; font-size:13px; margin:8px 0 4px; color:#bbb; }
  select, button, input { font-size:14px; padding:6px 8px; border-radius:6px; border:1px solid #555;
          background:#222; color:#eee; width:100%; }
  button { background:var(--red); border:none; cursor:pointer; font-weight:600; }
  button:hover { filter:brightness(1.1); }
  button.secondary { background:#444; }
  #status { font-size:15px; font-weight:600; min-height:22px; }
  #result { font-size:16px; color:#ffd36b; min-height:20px; margin-top:6px; }
  #moves { font-family: ui-monospace, monospace; font-size:12px; color:#cfcfcf; max-height:220px;
           overflow:auto; white-space:pre-wrap; line-height:1.5; }
  .row { display:flex; gap:8px; }
  .muted { color:#999; font-size:12px; }
</style>
</head>
<body>
<div class="wrap">
  <div>
    <h1>东风 · Dong Feng — Cờ tướng</h1>
    <svg id="board" width="540" height="600" viewBox="0 0 540 600"></svg>
  </div>
  <div class="panel">
    <div class="card">
      <div id="status">Đang tải…</div>
      <div id="result"></div>
    </div>
    <div class="card">
      <label>Đối thủ (engine)</label>
      <select id="engine">
        <option value="neural">Neural (model đã train)</option>
        <option value="random">Random (baseline)</option>
      </select>
      <label>Bạn cầm quân</label>
      <select id="human">
        <option value="red">Đỏ (đi trước)</option>
        <option value="black">Đen</option>
      </select>
      <label>Độ ngẫu nhiên (temperature): <span id="tval">0.0</span></label>
      <input type="range" id="temp" min="0" max="1.5" step="0.1" value="0"/>
      <div class="row" style="margin-top:10px;">
        <button id="new">Ván mới</button>
        <button id="undo" class="secondary" title="chưa hỗ trợ" disabled>Đi lại</button>
      </div>
    </div>
    <div class="card">
      <label>Nước đi (ICCS)</label>
      <div id="moves"></div>
    </div>
    <div class="muted">Bấm quân của bạn rồi bấm ô đích. Chấm xanh = nước hợp lệ.</div>
  </div>
</div>
<script>
const FILES = "abcdefghi";
const RED = {K:"帅",A:"仕",B:"相",N:"马",R:"车",C:"炮",P:"兵"};
const BLACK = {k:"将",a:"士",b:"象",n:"马",r:"车",c:"炮",p:"卒"};
const CELL=54, M=36;
const svg = document.getElementById("board");
let state=null, sel=null, busy=false;

function sq(col,rank){ return FILES[col]+rank; }
function px(col,rank){ return [M+col*CELL, M+(9-rank)*CELL]; }

function parseFen(fen){
  const rows = fen.split(" ")[0].split("/"); // row0 = rank9 (top)
  const occ = {}; // sq -> letter
  for(let i=0;i<10;i++){ const rank=9-i; let col=0;
    for(const ch of rows[i]){
      if(/\d/.test(ch)){ col+=parseInt(ch); }
      else { occ[sq(col,rank)]=ch; col++; }
    }
  }
  return occ;
}

function el(tag,attrs,text){ const e=document.createElementNS("http://www.w3.org/2000/svg",tag);
  for(const k in attrs) e.setAttribute(k,attrs[k]); if(text!=null) e.textContent=text; return e; }

function render(){
  svg.innerHTML="";
  // grid lines
  for(let r=0;r<10;r++){ const [x0,y]=px(0,r); const [x1]=px(8,r);
    svg.appendChild(el("line",{x1:x0,y1:y,x2:x1,y2:y,stroke:"var(--line)","stroke-width":1.4})); }
  for(let c=0;c<9;c++){
    if(c===0||c===8){ const [x,y0]=px(c,9); const [,y1]=px(c,0);
      svg.appendChild(el("line",{x1:x,y1:y0,x2:x,y2:y1,stroke:"var(--line)","stroke-width":1.4})); }
    else { // stop at the river (between rank5 and rank4)
      let a=px(c,9), b=px(c,5); svg.appendChild(el("line",{x1:a[0],y1:a[1],x2:b[0],y2:b[1],stroke:"var(--line)","stroke-width":1.4}));
      let d=px(c,4), e2=px(c,0); svg.appendChild(el("line",{x1:d[0],y1:d[1],x2:e2[0],y2:e2[1],stroke:"var(--line)","stroke-width":1.4})); }
  }
  // palaces (X)
  const palace=(r0,r1)=>{ let a=px(3,r0),b=px(5,r1); svg.appendChild(el("line",{x1:a[0],y1:a[1],x2:b[0],y2:b[1],stroke:"var(--line)","stroke-width":1.2}));
    let c=px(5,r0),d=px(3,r1); svg.appendChild(el("line",{x1:c[0],y1:c[1],x2:d[0],y2:d[1],stroke:"var(--line)","stroke-width":1.2})); };
  palace(9,7); palace(2,0);
  // river text
  const ry=M+4.5*CELL;
  svg.appendChild(el("text",{x:M+1.6*CELL,y:ry+6,"font-size":20,fill:"#7a4a1a","letter-spacing":6},"楚 河"));
  svg.appendChild(el("text",{x:M+5.4*CELL,y:ry+6,"font-size":20,fill:"#7a4a1a","letter-spacing":6},"漢 界"));

  if(!state) return;
  const occ=parseFen(state.fen);
  const last=state.last_move;
  if(last){ const [lx,ly]=px(FILES.indexOf(last[1][0]),parseInt(last[1][1]));
    svg.appendChild(el("circle",{cx:lx,cy:ly,r:CELL*0.46,fill:"none",stroke:"#3aa0ff","stroke-width":2})); }
  // legal dest dots for selected
  if(sel){ const dests=(state.legal||[]).filter(m=>m[0]===sel).map(m=>m[1]);
    for(const d of dests){ const [x,y]=px(FILES.indexOf(d[0]),parseInt(d[1]));
      const hit=occ[d]; svg.appendChild(el("circle",{cx:x,cy:y,r:hit?CELL*0.44:7,
        fill:hit?"none":"#2ecc71",stroke:hit?"#2ecc71":"none","stroke-width":3,opacity:0.85})); } }
  // pieces
  for(const s in occ){ const ch=occ[s]; const isRed=ch===ch.toUpperCase();
    const [x,y]=px(FILES.indexOf(s[0]),parseInt(s[1]));
    svg.appendChild(el("circle",{cx:x,cy:y,r:CELL*0.42,fill:"var(--cream)",
      stroke:isRed?"var(--red)":"var(--black)","stroke-width":s===sel?4:2}));
    svg.appendChild(el("text",{x:x,y:y+7,"font-size":24,"text-anchor":"middle",
      fill:isRed?"var(--red)":"var(--black)","font-weight":700},(isRed?RED:BLACK)[ch])); }
}

function nearest(evt){ const r=svg.getBoundingClientRect();
  const sx=(evt.clientX-r.left)*(540/r.width), sy=(evt.clientY-r.top)*(600/r.height);
  const col=Math.round((sx-M)/CELL), row=Math.round((sy-M)/CELL);
  if(col<0||col>8||row<0||row>9) return null; return sq(col,9-row); }

svg.addEventListener("click",async(evt)=>{
  if(busy||!state||state.result!=="ongoing"||state.turn!==state.human) return;
  const s=nearest(evt); if(!s) return;
  const occ=parseFen(state.fen); const ch=occ[s];
  const mine=ch && ((state.human==="red")===(ch===ch.toUpperCase()));
  if(sel && (state.legal||[]).some(m=>m[0]===sel&&m[1]===s)){ await move(sel,s); sel=null; }
  else if(mine){ sel=s; render(); }
  else { sel=null; render(); }
});

async function move(frm,to){ busy=true; setStatus("Máy đang nghĩ…");
  const r=await fetch("/api/move",{method:"POST",headers:{"Content-Type":"application/json"},
    body:JSON.stringify({from:frm,to:to})});
  const j=await r.json(); busy=false;
  if(j.error){ setStatus("⚠ "+j.error); return; }
  state=j.state; sel=null; render(); update(); }

async function refresh(){ state=await(await fetch("/api/state")).json(); render(); update(); }
async function newGame(){ busy=true; sel=null; setStatus("Đang tạo ván…");
  const body={engine:document.getElementById("engine").value,human:document.getElementById("human").value,
    temperature:parseFloat(document.getElementById("temp").value)};
  state=await(await fetch("/api/new",{method:"POST",headers:{"Content-Type":"application/json"},
    body:JSON.stringify(body)})).json(); busy=false; render(); update(); }

function setStatus(t){ document.getElementById("status").textContent=t; }
function update(){
  const names={red:"Đỏ",black:"Đen"};
  let s = state.result==="ongoing"
    ? "Lượt: "+names[state.turn]+(state.turn===state.human?" (bạn)":" (máy)")+(state.in_check?"  — CHIẾU!":"")
    : "Kết thúc";
  setStatus(s);
  const res={ongoing:"",red_win:"🔴 Đỏ thắng!",black_win:"⚫ Đen thắng!",draw:"Hòa"};
  document.getElementById("result").textContent=res[state.result]||"";
  let out=""; state.history.forEach((m,i)=>{ if(i%2===0) out+=(i/2+1)+". "; out+=m+(i%2===0?"  ":"\n"); });
  document.getElementById("moves").textContent=out;
}
document.getElementById("new").addEventListener("click",newGame);
document.getElementById("temp").addEventListener("input",e=>document.getElementById("tval").textContent=e.target.value);
refresh();
</script>
</body>
</html>
"""
