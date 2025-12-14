# webcam/services/hole_detector.py
import cv2
import json
from pathlib import Path
from typing import List, Tuple

BASELINE_PATH = Path(__file__).resolve().parent.parent / "table_holes_baseline.json"


def load_baseline_holes() -> Tuple[List[Tuple[int, int]], int]:
    """
    table_holes_baseline.json 에 저장된 홀 픽셀좌표 로드
    return: centers[(x,y)...], radius
    """
    if not BASELINE_PATH.exists():
        return [], 5

    data = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    radius = int(data.get("radius", 5))
    holes = data.get("holes", [])

    centers = [(int(h["x"]), int(h["y"])) for h in holes]
    return centers, radius


def detect_table_holes(frame, debug: bool = False):
    """
    수동으로 저장한 baseline 홀 좌표를 그대로 반환.
    return: centers, radii, debug_img
    """
    centers, r = load_baseline_holes()
    radii = [r for _ in centers]

    debug_img = frame.copy()
    if debug:
        for (x, y) in centers:
            cv2.circle(debug_img, (x, y), r, (0, 255, 0), 2)
            cv2.circle(debug_img, (x, y), 2, (0, 0, 255), 3)
        cv2.putText(
            debug_img,
            f"holes: {len(centers)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
        )

    return centers, radii, debug_img
