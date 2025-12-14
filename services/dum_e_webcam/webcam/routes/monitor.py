# webcam/routes/monitor.py
from fastapi import APIRouter
from fastapi.responses import HTMLResponse, JSONResponse

from webcam.monitor_state import get_state_snapshot

router = APIRouter()

_HTML = r"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>Dum-E Monitor</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <style>
    .glass { backdrop-filter: blur(10px); background: rgba(255,255,255,0.06); }
    .muted { color: rgba(255,255,255,0.65); }
    .chip { padding: 2px 10px; border-radius: 999px; font-size: 12px; }
    .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; }
  </style>
</head>
<body class="min-h-screen bg-gradient-to-b from-slate-950 via-slate-950 to-slate-900 text-slate-100">
  <div class="max-w-6xl mx-auto px-4 py-6">
    <div class="flex items-center justify-between gap-3">
      <div>
        <div class="text-2xl font-semibold tracking-tight">Dum-E Monitor</div>
        <div class="muted text-sm">실시간 상황 인지 / YOLO / ROI / GPT / Action</div>
      </div>
      <div class="flex items-center gap-2">
        <span id="pillCamera" class="chip bg-slate-800 text-slate-200">CAM: -</span>
        <span id="pillQueue" class="chip bg-slate-800 text-slate-200">QUEUE: -</span>
        <span id="pillGPT" class="chip bg-slate-800 text-slate-200">GPT: -</span>
      </div>
    </div>

    <!-- Banner -->
    <div id="banner" class="hidden mt-4 rounded-2xl glass border border-white/10 p-4">
      <div class="flex items-center justify-between">
        <div class="flex items-center gap-3">
          <div class="w-2.5 h-2.5 rounded-full bg-amber-400 animate-pulse"></div>
          <div class="font-medium">사진 속 상황 파악 중입니다…</div>
          <div id="bannerSub" class="muted text-sm">GPT 분석을 진행하고 있어요.</div>
        </div>
        <div id="bannerTimer" class="mono muted text-sm">-</div>
      </div>
    </div>

    <div class="mt-4 grid grid-cols-1 md:grid-cols-3 gap-4">
      <div class="md:col-span-2 space-y-4">
        <div class="rounded-2xl glass border border-white/10 p-4">
          <div class="flex items-center justify-between">
            <div class="font-semibold">Camera / YOLO</div>
            <div class="muted text-sm">last_frame: <span id="lastFrame">-</span></div>
          </div>

          <div class="mt-3 grid grid-cols-1 md:grid-cols-2 gap-3">
            <div class="rounded-xl bg-white/5 border border-white/10 p-3">
              <div class="muted text-xs">Detected</div>
              <div class="mt-1 text-lg font-semibold" id="yoloCls">-</div>
              <div class="mt-1 mono text-sm muted">conf=<span id="yoloConf">-</span></div>
            </div>

            <div class="rounded-xl bg-white/5 border border-white/10 p-3">
              <div class="muted text-xs">ROI</div>
              <div class="mt-1 text-lg font-semibold" id="roiIn">-</div>
              <div class="mt-1 mono text-sm muted">between_holes=<span id="betweenHoles">-</span></div>
            </div>

            <div class="rounded-xl bg-white/5 border border-white/10 p-3">
              <div class="muted text-xs">Center / BBox</div>
              <div class="mt-1 mono text-sm">
                center=<span id="yoloCenter" class="muted">-</span><br/>
                bbox=<span id="yoloBBox" class="muted">-</span>
              </div>
            </div>

            <div class="rounded-xl bg-white/5 border border-white/10 p-3">
              <div class="muted text-xs">Queue</div>
              <div class="mt-1 text-lg font-semibold">
                <span id="queueSize">-</span>
                <span class="muted text-sm"> / dropped </span>
                <span id="queueDropped">-</span>
              </div>
              <div class="mt-1 muted text-sm">큐가 꽉 차면 프레임이 drop 됩니다.</div>
            </div>
          </div>
        </div>

        <div class="rounded-2xl glass border border-white/10 p-4">
          <div class="flex items-center justify-between">
            <div class="font-semibold">Recent events</div>
            <div class="muted text-sm">최신 20개</div>
          </div>
          <div id="events" class="mt-3 space-y-2 max-h-[420px] overflow-auto pr-1"></div>
        </div>
      </div>

      <div class="space-y-4">
        <div class="rounded-2xl glass border border-white/10 p-4">
          <div class="font-semibold">GPT / Action</div>

          <div class="mt-3 rounded-xl bg-white/5 border border-white/10 p-3">
            <div class="muted text-xs">Risk</div>
            <div id="risk" class="mt-1 text-lg font-semibold">-</div>
          </div>

          <div class="mt-3 rounded-xl bg-white/5 border border-white/10 p-3">
            <div class="muted text-xs">Recommended action</div>
            <div id="action" class="mt-1 text-lg font-semibold">-</div>
          </div>

          <div class="mt-3 rounded-xl bg-white/5 border border-white/10 p-3">
            <div class="muted text-xs">Scene summary</div>
            <div id="summary" class="mt-2 text-sm leading-relaxed">-</div>
          </div>
        </div>

        <div class="rounded-2xl glass border border-white/10 p-4">
          <div class="font-semibold">Tips</div>
          <ul class="mt-2 list-disc list-inside muted text-sm space-y-1">
            <li>OpenCV 창은 별도로 유지해도 됩니다.</li>
            <li>폴링 주기(기본 300ms)를 늘리면 서버 부하가 줄어요.</li>
          </ul>
        </div>
      </div>
    </div>
  </div>

<script>
  const $ = (id) => document.getElementById(id);

  function tsToClock(ts) {
    if (!ts) return "-";
    const d = new Date(ts * 1000);
    const hh = String(d.getHours()).padStart(2,'0');
    const mm = String(d.getMinutes()).padStart(2,'0');
    const ss = String(d.getSeconds()).padStart(2,'0');
    return `${hh}:${mm}:${ss}`;
  }

  function fmt(val) {
    if (val === null || val === undefined) return "-";
    if (Array.isArray(val)) return JSON.stringify(val);
    if (typeof val === "object") return JSON.stringify(val);
    return String(val);
  }

  function pill(el, ok, text) {
    el.textContent = text;
    el.className = "chip " + (ok ? "bg-emerald-500/20 text-emerald-200 border border-emerald-500/30"
                                 : "bg-rose-500/20 text-rose-200 border border-rose-500/30");
  }

  // ✅ 폴링 중복 방지
  let _busy = false;

  async function tick() {
    if (_busy) return;
    _busy = true;

    let s;
    try {
      // ✅ 절대경로로 고정(서브경로/리버스프록시에도 안전)
      const r = await fetch("/api/status", { cache: "no-store" });
      s = await r.json();
    } catch (e) {
      pill($("pillCamera"), false, "CAM: ERR");
      _busy = false;
      return;
    }

    // Pills
    const camOk = !!s.camera_ok;
    if (s.camera_ok === null) {
      $("pillCamera").textContent = "CAM: -";
      $("pillCamera").className = "chip bg-slate-800 text-slate-200";
    } else {
      pill($("pillCamera"), camOk, "CAM: " + (camOk ? "OK" : "ERR"));
    }

    const qs = s.queue_size ?? 0;
    const qd = s.queue_dropped ?? 0;
    $("pillQueue").textContent = `QUEUE: ${qs} / drop ${qd}`;
    $("pillQueue").className = "chip bg-slate-800 text-slate-200";

    const analyzing = !!s.gpt_inference;
    $("pillGPT").textContent = "GPT: " + (analyzing ? "ANALYZING" : "IDLE");
    $("pillGPT").className = "chip " + (analyzing ? "bg-amber-500/20 text-amber-200 border border-amber-500/30"
                                                  : "bg-slate-800 text-slate-200");

    // Banner
    const banner = $("banner");
    if (analyzing) {
      banner.classList.remove("hidden");

      // ✅ 서브 문구도 상황에 맞게 바꿈
      const y = s.yolo || {};
      const obj = y.cls ? `${y.cls}` : "장면";
      $("bannerSub").textContent = `${obj} 포함 장면을 분석 중입니다.`;

      const since = s.gpt_inference_since;
      if (since) {
        const sec = Math.max(0, (Date.now()/1000 - since));
        $("bannerTimer").textContent = `${sec.toFixed(1)}s`;
      } else {
        $("bannerTimer").textContent = "-";
      }
    } else {
      banner.classList.add("hidden");
      $("bannerSub").textContent = "GPT 분석을 진행하고 있어요.";
      $("bannerTimer").textContent = "-";
    }

    // Camera / YOLO
    $("lastFrame").textContent = tsToClock(s.last_frame_ts);

    const y = s.yolo || {};
    $("yoloCls").textContent = y.cls ? y.cls : "-";
    $("yoloConf").textContent = (y.conf !== null && y.conf !== undefined) ? y.conf : "-";
    $("yoloCenter").textContent = fmt(y.center);
    $("yoloBBox").textContent = fmt(y.bbox);

    const inroi = y.in_table_roi;
    $("roiIn").textContent = (inroi === null || inroi === undefined) ? "-" : (inroi ? "INSIDE" : "OUTSIDE");
    $("betweenHoles").textContent = fmt(y.between_holes);

    $("queueSize").textContent = qs;
    $("queueDropped").textContent = qd;

    // GPT / Action
    const g = s.gpt || {};
    const a = s.action || {};
    $("risk").textContent = g.risk_level || "-";
    $("action").textContent = a.recommended_action || "-";
    $("summary").textContent = g.scene_summary || "-";

    // Events
    const evs = s.events || [];
    const box = $("events");
    box.innerHTML = "";
    evs.slice(0, 20).forEach(ev => {
      const row = document.createElement("div");
      row.className = "rounded-xl bg-white/5 border border-white/10 px-3 py-2";
      row.innerHTML = `
        <div class="mono muted text-xs">${tsToClock(ev.t)} <span class="text-slate-300">[${ev.tag}]</span></div>
        <div class="mt-1 text-sm">${(ev.msg || "").toString()}</div>
      `;
      box.appendChild(row);
    });

    _busy = false;
  }

  tick();
  setInterval(tick, 300);
</script>
</body>
</html>
"""

@router.get("/monitor", response_class=HTMLResponse)
def monitor_page():
    return HTMLResponse(_HTML)

@router.get("/api/status")
def api_status():
    return JSONResponse(get_state_snapshot())
