# services/audio_io/app/webcam.py
from __future__ import annotations
import os
import base64
from pathlib import Path
from typing import Optional, Tuple

import cv2

DEFAULT_INDEX = int(os.getenv("DUM_E_WEBCAM_INDEX", "6"))
DEFAULT_WIDTH = int(os.getenv("DUM_E_WEBCAM_WIDTH", "640"))
DEFAULT_HEIGHT = int(os.getenv("DUM_E_WEBCAM_HEIGHT", "480"))

# 디버깅용 저장 위치
DEFAULT_SAVE_PATH = Path(os.getenv("DUM_E_WEBCAM_SAVE_PATH", "/tmp/dum_e_webcam_latest.jpg"))


def capture_webcam_jpeg(
    index: int = DEFAULT_INDEX,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    save_path: Optional[Path] = DEFAULT_SAVE_PATH,
) -> Tuple[bytes, Optional[str]]:
    """
    USB 웹캠에서 프레임 1장을 캡쳐해 JPEG bytes로 반환.
    실패하면 예외를 던진다.
    return: (jpeg_bytes, saved_file_path_str_or_none)
    """
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"Webcam open failed: index={index}")

    # 해상도는 카메라가 지원 안 하면 무시될 수 있음
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))

    # 첫 프레임이 어둡거나 stale인 경우가 있어서 몇 번 워밍업
    frame = None
    for _ in range(5):
        ok, f = cap.read()
        if ok and f is not None:
            frame = f

    cap.release()

    if frame is None:
        raise RuntimeError("Webcam read failed: no valid frame")

    ok, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        raise RuntimeError("JPEG encode failed")

    jpg_bytes = jpg.tobytes()

    saved = None
    if save_path is not None:
        try:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            save_path.write_bytes(jpg_bytes)
            saved = str(save_path)
        except Exception:
            saved = None

    return jpg_bytes, saved


def jpeg_bytes_to_data_url(jpg_bytes: bytes) -> str:
    b64 = base64.b64encode(jpg_bytes).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"
