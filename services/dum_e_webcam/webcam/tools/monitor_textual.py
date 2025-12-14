import json, time, urllib.request
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Header, Footer, Static
from textual.reactive import reactive

URL = "http://127.0.0.1:8000/api/status"

def fetch():
    with urllib.request.urlopen(URL, timeout=2) as r:
        return json.loads(r.read().decode())

def ts(t):
    if not t:
        return "-"
    return time.strftime("%H:%M:%S", time.localtime(t))

class Box(Static):
    pass

class MonitorApp(App):
    CSS = """
    Screen { background: #0b1020; color: #e8f0ff; }
    .card { border: round #2b355a; padding: 1 2; margin: 1; }
    .title { text-style: bold; color: #b8c7ff; }
    .ok { color: #7CFC98; }
    .warn { color: #FFD36A; }
    .bad { color: #FF6B6B; }
    .muted { color: #9fb0ff; }

    /* ✅ 오른쪽 레이아웃 고정 */
    #events_card { height: 1fr; }
    #conn_card   { height: 9; }
    """


    status = reactive({})

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal():
            with Vertical():
                self.yolo = Box(classes="card")
                self.gpt = Box(classes="card")
                yield self.yolo
                yield self.gpt

            with Vertical():
                self.events = Box(classes="card")
                self.conn = Box(classes="card")   # ✅ 오른쪽 하단 연결상태 박스
                yield self.events
                yield self.conn
        yield Footer()

    async def on_mount(self):
        self._last_ok_ts = None
        self._last_err = None
        self._last_latency_ms = None
        self.set_interval(0.3, self.tick)

    def render_yolo(self, s):
        y = s.get("yolo") or {}
        robot_xy = s.get("robot_target_xy")
        cam_ok = s.get("camera_ok")
        qsz = s.get("queue_size")
        qdp = s.get("queue_dropped")

        cam_txt = "-"
        if cam_ok is True:
            cam_txt = "[green]OK[/green]"
        elif cam_ok is False:
            cam_txt = "[red]ERR[/red]"

        return (
            "[b]YOLO / ROI / RobotXY[/b]\n"
            f"camera: {cam_txt}  last_frame: {ts(s.get('last_frame_ts'))}\n"
            f"queue: size={qsz} dropped={qdp}\n\n"
            f"cls={y.get('cls')}\n"
            f"conf={y.get('conf')}\n"
            f"center={y.get('center')}\n"
            f"bbox={y.get('bbox')}\n"
            f"in_table_roi={y.get('in_table_roi')}\n"
            f"between_holes={y.get('between_holes')}\n\n"
            f"robot_target_xy={robot_xy}\n"
        )

    def render_gpt(self, s):
        g = s.get("gpt") or {}
        a = s.get("action") or {}
        inf = bool(s.get("gpt_inference"))

        risk = (g.get("risk_level") or "").lower()
        if risk == "high":
            risk_txt = "[bold red]HIGH[/bold red]"
        elif risk == "medium":
            risk_txt = "[bold yellow]MEDIUM[/bold yellow]"
        elif risk == "low":
            risk_txt = "[bold green]LOW[/bold green]"
        else:
            risk_txt = "-"

        banner = "\n[cyan]사진 속 상황 파악 중입니다...[/cyan]\n" if inf else ""

        return (
            "[b]GPT / Action[/b]\n"
            f"risk={risk_txt}\n"
            f"action={a.get('recommended_action')}\n"
            f"inference={'[yellow]ANALYZING[/yellow]' if inf else 'IDLE'}\n"
            f"{banner}\n"
            f"summary:\n{g.get('scene_summary')}\n"
        )

    def render_events(self, s):
        # ✅ 여기서 “누적”이 아니라 항상 최신 25개로 덮어씀
        evs = (s.get("events") or [])[:25]
        lines = ["[b]Recent Events[/b]"]
        for e in evs:
            lines.append(f"{ts(e.get('t'))} [{e.get('tag')}] {e.get('msg')}")
        return "\n".join(lines)

    def render_conn(self):
        # ✅ 오른쪽 하단 연결상태 표시
        if self._last_ok_ts is None:
            ok_line = "[red]API: NOT CONNECTED[/red]"
        else:
            ok_line = f"[green]API: OK[/green]  last_ok={ts(self._last_ok_ts)}  latency={self._last_latency_ms}ms"

        err_line = ""
        if self._last_err:
            err_line = f"\n[red]last_error:[/red] {self._last_err}"

        return (
            "[b]Connection[/b]\n"
            f"{ok_line}"
            f"{err_line}\n"
        )

    def tick(self):
        t0 = time.time()
        try:
            s = fetch()
            self._last_ok_ts = time.time()
            self._last_latency_ms = int((self._last_ok_ts - t0) * 1000)
            self._last_err = None
        except Exception as e:
            # ✅ 실패해도 UI는 유지
            s = self.status or {}
            self._last_err = str(e)[:160]

        self.status = s
        self.yolo.update(self.render_yolo(s))
        self.gpt.update(self.render_gpt(s))
        self.events.update(self.render_events(s))
        self.conn.update(self.render_conn())

if __name__ == "__main__":
    MonitorApp().run()
