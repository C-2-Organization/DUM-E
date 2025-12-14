# webcam/services/location_mapper.py
import json
from pathlib import Path
from typing import Dict, Tuple, Optional, List

BASELINE_PATH = Path(__file__).resolve().parent.parent / "/home/ilhoon/DUM-E/services/dum_e_webcam/webcam/table_holes_baseline.json"

RobotXY = Tuple[float, float]
CellCorners = Tuple[int, int, int, int]

def _load_hole_map() -> Dict[int, RobotXY]:
    """
    table_holes_baseline.json의 holes에서 id -> (robot_x, robot_y) 맵 생성
    """
    data = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    holes = data.get("holes", [])
    m: Dict[int, RobotXY] = {}

    for h in holes:
        hid = int(h["id"])
        rx = float(h["robot"]["x"])
        ry = float(h["robot"]["y"])
        m[hid] = (rx, ry)
    return m

HOLE_ROBOT_MAP = _load_hole_map()

def cell_to_robot_xy(corners: CellCorners) -> Optional[RobotXY]:
    """
    셀 4개 홀(TL,TR,BL,BR)의 robot 좌표 평균(=셀 중앙 목표점) 반환
    """
    tl, tr, bl, br = corners
    pts: List[RobotXY] = []
    for hid in (tl, tr, bl, br):
        if hid not in HOLE_ROBOT_MAP:
            return None
        pts.append(HOLE_ROBOT_MAP[hid])

    x = sum(p[0] for p in pts) / 4.0
    y = sum(p[1] for p in pts) / 4.0
    return (x, y)

from typing import Optional, Tuple, Any

def between_holes_to_robot_xy(between_holes: Any) -> Optional[Tuple[float, float]]:
    """
    between_holes(예: [tl,tr,bl,br])로부터 로봇 XY 추정.
    내부 매핑 방식이 다를 수 있어 try/fallback 구조로 둠.
    """
    if not between_holes or between_holes == "skip":
        return None

    # between_holes = [[r,c], [r,c], [r,c], [r,c]] 같은 형태라고 가정(워핑 셀 코너)
    try:
        tl, tr, bl, br = between_holes
    except Exception:
        return None

    # 1) 프로젝트에 이미 있는 cell_to_robot_xy를 최대한 활용
    try:
        # 케이스1: cell_to_robot_xy(tl,tr,bl,br) 를 기대
        return cell_to_robot_xy(tl, tr, bl, br)
    except Exception:
        pass

    try:
        # 케이스2: cell_to_robot_xy(cell_row, cell_col) 를 기대(가운데 셀)
        r = int(round((tl[0] + tr[0] + bl[0] + br[0]) / 4.0))
        c = int(round((tl[1] + tr[1] + bl[1] + br[1]) / 4.0))
        return cell_to_robot_xy(r, c)
    except Exception:
        return None
