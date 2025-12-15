from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Any
import json
import math


# -----------------------------
# Data model
# -----------------------------
@dataclass
class Hole:
    """
    id: 1..N (detect_table_holes()가 그리는 순서와 매칭되는 번호)
    img: (x_px, y_px)
    robot: (x_mm, y_mm)  # ✅ 헴 시스템: mm
    """
    id: int
    img: Tuple[float, float]
    robot: Tuple[float, float]


def _dist2(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return dx * dx + dy * dy


# -----------------------------
# Baseline loader
# -----------------------------
def load_holes_baseline(json_path: str) -> Dict[int, Tuple[float, float]]:
    """
    table_holes_baseline.json 로딩

    기대 포맷(예시):
    {
      "1": {"x": 400.0, "y": -180.0},
      "2": {"x": 410.0, "y": -175.0},
      ...
    }

    return:
      { 1: (x_mm, y_mm), 2: (x_mm, y_mm), ... }
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    out: Dict[int, Tuple[float, float]] = {}
    if isinstance(data, dict):
        for k, v in data.items():
            try:
                hid = int(k)
            except Exception:
                continue

            if isinstance(v, dict):
                x = float(v.get("x"))
                y = float(v.get("y"))
                out[hid] = (x, y)
    return out


def build_holes_from_detected_and_baseline(
    holes_img: List[Tuple[float, float]],
    baseline_robot_xy: Dict[int, Tuple[float, float]],
) -> List[Hole]:
    """
    holes_img: detect_table_holes() 결과 [(x_px,y_px), ...]
    baseline_robot_xy: load_holes_baseline() 결과 {id: (x_mm,y_mm)}

    return: Hole 리스트 (baseline에 있는 id만 robot 좌표 부여)
    """
    holes: List[Hole] = []
    for i, (hx, hy) in enumerate(holes_img, start=1):
        robot = baseline_robot_xy.get(i)
        if robot is None:
            # baseline에 해당 id가 없으면 스킵(또는 (0,0) 넣어도 되지만 위험)
            continue
        holes.append(Hole(id=i, img=(float(hx), float(hy)), robot=(float(robot[0]), float(robot[1]))))
    return holes


# -----------------------------
# KNN estimators (헴 코드 기반)
# -----------------------------
def estimate_robot_xy_knn_mean(
    cx: float,
    cy: float,
    holes: List[Hole],
    k: int = 3,
) -> Optional[Dict[str, Any]]:
    if not holes:
        return None

    k = max(1, min(int(k), len(holes)))
    p = (float(cx), float(cy))

    ranked = sorted([(h, math.sqrt(_dist2(p, h.img))) for h in holes], key=lambda x: x[1])
    nn = ranked[:k]

    rx = sum(h.robot[0] for h, _ in nn) / k
    ry = sum(h.robot[1] for h, _ in nn) / k

    return {
        "robot_xy": (round(rx, 2), round(ry, 2)),
        "neighbor_ids": [h.id for h, _ in nn],
        "pix_dists": [round(d, 2) for _, d in nn],
        "k": k,
    }


def estimate_robot_xy_knn_idw(
    cx: float,
    cy: float,
    holes: List[Hole],
    k: int = 3,
    power: float = 1.0,
) -> Optional[Dict[str, Any]]:
    if not holes:
        return None

    k = max(1, min(int(k), len(holes)))
    p = (float(cx), float(cy))

    ranked = sorted([(h, math.sqrt(_dist2(p, h.img))) for h in holes], key=lambda x: x[1])
    nn = ranked[:k]

    eps = 1e-6
    ws = []
    for _, d in nn:
        w = 1.0 / ((d + eps) ** power)
        ws.append(w)

    s = sum(ws) if ws else 1.0
    ws = [w / s for w in ws]

    rx = 0.0
    ry = 0.0
    for (h, _d), w in zip(nn, ws):
        rx += w * h.robot[0]
        ry += w * h.robot[1]

    return {
        "robot_xy": (round(rx, 2), round(ry, 2)),
        "neighbor_ids": [h.id for h, _ in nn],
        "weights": [round(w, 3) for w in ws],
        "pix_dists": [round(d, 2) for _, d in nn],
        "k": k,
    }


# -----------------------------
# Compatibility wrapper (situation_worker가 쓰기 좋게)
# -----------------------------
def estimate_robot_xy_from_center(
    cx: float,
    cy: float,
    holes_img: List[Tuple[float, float]],
    baseline_robot_xy: Dict[int, Tuple[float, float]],
    k: int = 3,
    method: str = "idw",   # "mean" or "idw"
    power: float = 1.0,
) -> Optional[Dict[str, Any]]:
    """
    situation_worker에서 바로 쓰게 만든 래퍼.

    return 예시:
      {
        "robot_xy": (x_mm, y_mm),
        "neighbor_ids": [...],
        ...
      }
    """
    holes = build_holes_from_detected_and_baseline(holes_img, baseline_robot_xy)
    if not holes:
        return None

    m = (method or "idw").lower()
    if m in ("mean", "avg", "knn_mean"):
        return estimate_robot_xy_knn_mean(cx, cy, holes, k=k)

    # default: idw
    return estimate_robot_xy_knn_idw(cx, cy, holes, k=k, power=power)
