#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"   # dum_e_webcam 루트 고정

UVICORN_CMD='uvicorn webcam.main:webcam --reload'
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

# 서버 뜰 시간 조금 주기
sleep 1

# ✅ UI를 "새 터미널"에서 실행
if command -v gnome-terminal >/dev/null 2>&1; then
  echo "[RUN] (new terminal) $UI_CMD"
  gnome-terminal --title="Dum-E Monitor UI" -- bash -lc "$UI_CMD; echo; echo '[UI] exited. press enter to close.'; read"
else
  echo "[WARN] gnome-terminal not found -> running in current terminal"
  bash -lc "$UI_CMD"
fi

# 현재 터미널은 uvicorn 로그를 계속 보여주게 유지
echo "[INFO] UI is running in another terminal. This terminal shows uvicorn logs."
wait "$UV_PID"
