#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# ✅ 중요: --reload 제거 (ROS + 스레드 + uvicorn reload 조합이 꼬임)
UVICORN_CMD='uvicorn webcam.main:webcam'
UI_CMD='python3 webcam/tools/monitor_textual.py'

echo "[RUN] $UVICORN_CMD"
bash -lc "$UVICORN_CMD" &
UV_PID=$!

cleanup() {
  echo ""
  echo "[STOP] stopping uvicorn (pid=$UV_PID)"
  kill "$UV_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

sleep 1

if command -v gnome-terminal >/dev/null 2>&1; then
  echo "[RUN] (new terminal) $UI_CMD"
  gnome-terminal --title="Dum-E Monitor UI" -- bash -lc "$UI_CMD; echo; echo '[UI] exited. press enter to close.'; read"
else
  echo "[WARN] gnome-terminal not found -> running in current terminal"
  bash -lc "$UI_CMD"
fi

echo "[INFO] UI is running in another terminal. This terminal shows uvicorn logs."
wait "$UV_PID"
