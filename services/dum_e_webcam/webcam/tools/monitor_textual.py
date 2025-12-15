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

def ko_obj(cls_name: str | None) -> str:
    if not cls_name:
        return "알 수 없음"
    mapping = {
        "scissors": "가위",
        "chair": "의자",
        "bottle": "병",
        "cup": "컵",
        "person": "사람",
        "hand": "손",
        "keyboard": "키보드",
        "mouse": "마우스",
        "phone": "휴대폰",
        "remote": "리모컨",
        "knife": "칼",
        "box_cutter": "커터칼",
        "hammer": "망치",
    }
    return mapping.get(cls_name, cls_name)

class Box(Static):
    pass

class MonitorApp(App):
    CSS = """
    Screen { background: #0b1020; color: #e8f0ff; }
    .card { border: round #2b355a; padding: 1 2; margin: 1; }
    .title { text-style: bold; color: #b8c7ff; }

    #events_card { height: 12; }
    #conn_card   { height: 9; }

    #roi_card   { height: 1fr; }   /* ✅ ROI도 여러개 표시하려면 늘려야 함 */
    #yolo_card  { height: 1fr; }
    #robot_card { height: 7; }
    #gpt_card   { height: 1fr; }
    """

    status = reactive({})

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal():
            with Vertical():
                self.roi = Box(id="roi_card", classes="card")
                self.yolo = Box(id="yolo_card", classes="card")
                self.robot = Box(id="robot_card", classes="card")
                self.gpt = Box(id="gpt_card", classes="card")
                yield self.roi
                yield self.yolo
                yield self.robot
                yield self.gpt

            with Vertical():
                self.events = Box(id="events_card", classes="card")
                self.conn = Box(id="conn_card", classes="card")
                yield self.events
                yield self.conn
        yield Footer()

    async def on_mount(self):
        self._last_ok_ts = None
        self._last_err = None
        self._last_latency_ms = None
        self.set_interval(0.3, self.tick)

    # =========================
    # ROI (멀티 물체)
    # =========================
    def render_roi(self, s):
        cam_ok = s.get("camera_ok")
        qsz = s.get("queue_size")
        qdp = s.get("queue_dropped")

        if cam_ok is True:
            cam_txt = "[green]정상[/green]"
        elif cam_ok is False:
            cam_txt = "[red]오류[/red]"
        else:
            cam_txt = "-"

        y = (s.get("yolo") or {})
        confirmed = y.get("confirmed") or []

        lines = [
            "[b]ROI[/b]",
            f"카메라: {cam_txt}  마지막 프레임: {ts(s.get('last_frame_ts'))}",
            f"큐 상태: 쌓인 큐 개수={qsz}  버린 큐 개수={qdp}",
            "",
        ]

        if confirmed:
            lines.append(f"확정 물체 수: {len(confirmed)}")
            MAX_SHOW = 10

            for i, d in enumerate(confirmed[:MAX_SHOW], 1):
                cls_ko = ko_obj(d.get("cls_name"))
                in_roi = d.get("in_table_roi")
                between = d.get("between_holes") or "-"
                center = d.get("center")

                # in_roi가 None일 수도 있으니까 표시 방어
                if in_roi is True:
                    in_txt = "[green]True[/green]"
                elif in_roi is False:
                    in_txt = "[red]False[/red]"
                else:
                    in_txt = "-"

                lines.append(f"{i:02d}) {cls_ko} center={center}")
                lines.append(f"    테이블 영역 여부: {in_txt} / 위치 판정: {between}")
        else:
            lines.append("ROI 안에 물체가 없습니다")

        return "\n".join(lines)

    # =========================
    # YOLO (멀티 물체)
    # =========================
    def render_yolo(self, s):
        y = s.get("yolo") or {}
        confirmed = y.get("confirmed") or []

        lines = ["[b]YOLO[/b]"]

        if confirmed:
            lines.append(f"확정 물체 수: {len(confirmed)}")
            MAX_SHOW = 10

            for i, d in enumerate(confirmed[:MAX_SHOW], 1):
                cls_ko = ko_obj(d.get("cls_name"))
                conf = d.get("conf")
                center = d.get("center")
                hit = d.get("hit")
                tid = d.get("track_id")
                lines.append(f"{i:02d}) {cls_ko} conf={conf} center={center} hit={hit} id={tid}")

            if len(confirmed) > MAX_SHOW:
                lines.append(f"... +{len(confirmed) - MAX_SHOW}개 더 있음")
        else:
            lines.append("확정된 물체 없음")

        return "\n".join(lines)

    def render_robot(self, s):
        robot_xy = s.get("robot_target_xy")
        return (
            "[b]RobotXY[/b]\n"
            f"로봇 목표 좌표: {robot_xy}\n"
        )

    def render_gpt(self, s):
        g = s.get("gpt") or {}
        a = s.get("action") or {}
        inf = bool(s.get("gpt_inference"))

        risk = (g.get("risk_level") or "").lower()
        if risk == "high":
            risk_txt = "[bold red]높음[/bold red]"
        elif risk == "medium":
            risk_txt = "[bold yellow]중간[/bold yellow]"
        elif risk == "low":
            risk_txt = "[bold green]낮음[/bold green]"
        else:
            risk_txt = "-"

        banner = "\n[cyan]사진 속 상황 파악 중입니다...[/cyan]\n" if inf else ""

        return (
            "[b]GPT / Action[/b]\n"
            f"위험도: {risk_txt}\n"
            f"추천 행동: {a.get('recommended_action')}\n"
            f"추론 상태: {'[yellow]분석 중[/yellow]' if inf else '대기'}\n"
            f"{banner}\n"
            f"요약:\n{g.get('scene_summary')}\n"
        )

    def render_events(self, s):
        evs = (s.get("events") or [])[:25]
        lines = ["[b]최근 이벤트[/b]"]
        for e in evs:
            lines.append(f"{ts(e.get('t'))} [{e.get('tag')}] {e.get('msg')}")
        return "\n".join(lines)

    def render_conn(self):
        if self._last_ok_ts is None:
            ok_line = "[red]API: 연결 안 됨[/red]"
        else:
            ok_line = f"[green]API: 연결됨[/green]  마지막 응답: {ts(self._last_ok_ts)}  지연={self._last_latency_ms}ms"

        err_line = ""
        if self._last_err:
            err_line = f"\n[red]마지막 오류:[/red] {self._last_err}"

        return (
            "[b]연결 상태[/b]\n"
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
            s = self.status or {}
            self._last_err = str(e)[:160]

        self.status = s
        self.roi.update(self.render_roi(s))
        self.yolo.update(self.render_yolo(s))
        self.robot.update(self.render_robot(s))
        self.gpt.update(self.render_gpt(s))
        self.events.update(self.render_events(s))
        self.conn.update(self.render_conn())

if __name__ == "__main__":
    MonitorApp().run()
