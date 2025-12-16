from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Any
import json
import math


# =============================
# Data model
# =============================
@dataclass
class Hole:
    """
    id      : hole id (1..N)
    img     : (x_px, y_px)
    robot   : (x_mm, y_mm)
    """
    id: int
    img: Tuple[float, float]
    robot: Tuple[float, float]


def _dist2(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return dx * dx + dy * dy


# =============================
# Baseline loader (🔥 핵심 수정)
# =============================
def load_holes_baseline(json_path: str) -> List[Hole]:
    """
    헴이 쓰는 table_holes_baseline.json 로딩

    기대 포맷:
    {
      "radius": 5,
      "holes": [
        {
          "id": 1,
          "x": 191,
          "y": 336,
          "robot": { "x": 723.3, "y": -230.0 }
        },
        ...
      ]
    }

    return:
      List[Hole]
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    holes_data = data.get("holes", [])
    holes: List[Hole] = []

    for h in holes_data:
        try:
            hid = int(h["id"])
            ix = float(h["x"])
            iy = float(h["y"])
            rx = float(h["robot"]["x"])
            ry = float(h["robot"]["y"])
        except Exception:
            continue

        holes.append(
            Hole(
                id=hid,
                img=(ix, iy),
                robot=(rx, ry),
            )
        )

    return holes


# =============================
# KNN mean estimator (정답 로직)
# =============================
def estimate_robot_xy_knn_mean(
    cx: float,
    cy: float,
    holes: List[Hole],
    k: int = 2,
) -> Optional[Dict[str, Any]]:
    """
    가장 가까운 k개 홀의 robot 좌표를 단순 평균
    - 격자 구조 전용
    - 헴 요구사항: 2개면 /2, 3개면 /3
    """
    if not holes:
        return None

    p = (float(cx), float(cy))
    k = max(1, min(int(k), len(holes)))

    ranked = sorted(
        [(h, math.sqrt(_dist2(p, h.img))) for h in holes],
        key=lambda x: x[1],
    )

    nn = ranked[:k]

    rx = sum(h.robot[0] for h, _ in nn) / k
    ry = sum(h.robot[1] for h, _ in nn) / k

    return {
        "robot_xy": (round(rx, 2), round(ry, 2)),
        "neighbor_ids": [h.id for h, _ in nn],
        "pix_dists": [round(d, 2) for _, d in nn],
        "k": k,
    }


# =============================
# situation_worker용 래퍼
# =============================
def estimate_robot_xy_from_center(
    cx: float,
    cy: float,
    baseline_holes: List[Hole],
    k: int = 2,
) -> Optional[Dict[str, Any]]:
    """
    situation_worker에서 바로 호출용

    return:
      {
        "robot_xy": (x_mm, y_mm),
        "neighbor_ids": [...],
        "k": k,
      }
    """
    return estimate_robot_xy_knn_mean(cx, cy, baseline_holes, k=k)
