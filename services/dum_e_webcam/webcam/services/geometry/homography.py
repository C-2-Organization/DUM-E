# webcam/services/homography.py
import json
import cv2
import numpy as np
from pathlib import Path

ROI_JSON_PATH = Path(__file__).resolve().parent.parent / "table_roi.json"

def load_roi_polygon():
    """
    table_roi.json의 polygon(4점)을 로드해서 (4,2) float32 반환
    """
    if not ROI_JSON_PATH.exists():
        raise FileNotFoundError(f"ROI json not found: {ROI_JSON_PATH}")

    data = json.loads(ROI_JSON_PATH.read_text(encoding="utf-8"))
    poly = data.get("polygon", [])
    if len(poly) != 4:
        raise ValueError("polygon은 반드시 4점이어야 합니다 (현재 points=%d)" % len(poly))

    pts = np.array([[p["x"], p["y"]] for p in poly], dtype=np.float32)
    return pts

def order_points(pts: np.ndarray) -> np.ndarray:
    """
    4점을 TL, TR, BR, BL 순으로 정렬
    """
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)          # x+y
    diff = np.diff(pts, axis=1)  # x-y

    rect[0] = pts[np.argmin(s)]     # TL
    rect[2] = pts[np.argmax(s)]     # BR
    rect[1] = pts[np.argmin(diff)]  # TR
    rect[3] = pts[np.argmax(diff)]  # BL
    return rect

def compute_homography(dst_w: int = 800, dst_h: int = 600):
    """
    ROI 4점 -> top-down(직사각형)으로 펴는 호모그래피 행렬 M 반환
    """
    src = load_roi_polygon()
    src = order_points(src)

    dst = np.array(
        [[0, 0],
         [dst_w - 1, 0],
         [dst_w - 1, dst_h - 1],
         [0, dst_h - 1]],
        dtype=np.float32
    )

    M = cv2.getPerspectiveTransform(src, dst)
    return M, (dst_w, dst_h), src, dst

def warp_frame(frame_bgr, M, dst_size):
    w, h = dst_size
    return cv2.warpPerspective(frame_bgr, M, (w, h))

def warp_point(x: int, y: int, M):
    """
    원본 픽셀 (x,y) -> 워핑 좌표 (u,v)
    """
    p = np.array([[[float(x), float(y)]]], dtype=np.float32)
    q = cv2.perspectiveTransform(p, M)[0][0]
    return int(q[0]), int(q[1])
