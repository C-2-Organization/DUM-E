import json
import cv2
import numpy as np
from pathlib import Path

ROI_PATH = Path(__file__).resolve().parent.parent / "table_roi.json"

def load_table_polygon():
    if not ROI_PATH.exists():
        return None
    data = json.loads(ROI_PATH.read_text(encoding="utf-8"))
    poly = data.get("polygon", [])
    if len(poly) < 3:
        return None
    pts = np.array([[p["x"], p["y"]] for p in poly], dtype=np.int32)
    return pts

def point_in_table(px: int, py: int) -> bool:
    pts = load_table_polygon()
    if pts is None:
        return False
    # pointPolygonTest: inside>0, on edge=0, outside<0
    res = cv2.pointPolygonTest(pts, (float(px), float(py)), False)
    return res >= 0
