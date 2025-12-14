# webcam/services/hole_grid_warp.py
import json
from pathlib import Path
from typing import Dict, Tuple, List, Optional

from webcam.services.geometry.homography import compute_homography, warp_point

BASELINE_PATH = Path(__file__).resolve().parent.parent / "table_holes_baseline.json"

CellCorners = Tuple[int, int, int, int]

def load_holes():
    data = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    return data["holes"]

def build_grid_by_robot(holes: List[dict]) -> Dict[Tuple[int, int], dict]:
    """
    robot 좌표로 4x4 grid 구성 (row: y 작은->큰, col: x 큰->작은)
    """
    xs = sorted({float(h["robot"]["x"]) for h in holes}, reverse=True)
    ys = sorted({float(h["robot"]["y"]) for h in holes})

    grid: Dict[Tuple[int,int], dict] = {}
    for h in holes:
        rx = float(h["robot"]["x"])
        ry = float(h["robot"]["y"])
        col = xs.index(rx)
        row = ys.index(ry)
        grid[(row, col)] = h
    return grid

def warped_cell_list() -> List[Tuple[CellCorners, Tuple[int,int,int,int]]]:
    """
    호모그래피로 홀 좌표를 워핑한 뒤 3x3 셀 bbox 생성
    return: [(corners_ids, (minx,maxx,miny,maxy)), ...]
    """
    M, dst_size, _, _ = compute_homography(dst_w=800, dst_h=600)

    holes = load_holes()
    grid = build_grid_by_robot(holes)

    cells = []
    for row in range(3):
        for col in range(3):
            tl = grid[(row, col)]
            tr = grid[(row, col+1)]
            bl = grid[(row+1, col)]
            br = grid[(row+1, col+1)]

            corners = (tl["id"], tr["id"], bl["id"], br["id"])

            # 원본 홀 픽셀 -> 워핑 홀 픽셀
            tl_u, tl_v = warp_point(tl["x"], tl["y"], M)
            tr_u, tr_v = warp_point(tr["x"], tr["y"], M)
            bl_u, bl_v = warp_point(bl["x"], bl["y"], M)
            br_u, br_v = warp_point(br["x"], br["y"], M)

            xs = [tl_u, tr_u, bl_u, br_u]
            ys = [tl_v, tr_v, bl_v, br_v]
            minx, maxx = min(xs), max(xs)
            miny, maxy = min(ys), max(ys)

            cells.append((corners, (minx, maxx, miny, maxy)))

    return cells, M

def locate_point_to_cell_warped(x: int, y: int) -> Optional[CellCorners]:
    """
    원본 픽셀(x,y)을 워핑해서, 워핑 셀 bbox 기준으로 between_holes 반환
    """
    cells, M = warped_cell_list()
    u, v = warp_point(x, y, M)

    for corners, (minx, maxx, miny, maxy) in cells:
        if minx <= u <= maxx and miny <= v <= maxy:
            return corners
    return None
