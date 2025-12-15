import cv2
import json
from pathlib import Path
from typing import List, Tuple

BASELINE_PATH = Path(__file__).resolve().parent.parent / "/home/ilhoon/DUM-E/services/dum_e_webcam/webcam/table_holes_baseline.json"


def load_baseline_holes() -> Tuple[List[Tuple[int, int, int]], int]:
    """
    table_holes_baseline.json 에 저장된 홀 픽셀좌표 로드
    return: holes[(id,x,y)...], radius
    """
    if not BASELINE_PATH.exists():
        return [], 5

    data = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    radius = int(data.get("radius", 5))
    holes = data.get("holes", [])

    # ✅ id 포함해서 로드 (id 없으면 0)
    hole_list: List[Tuple[int, int, int]] = []
    for h in holes:
        hid = int(h.get("id", 0))
        x = int(h["x"])
        y = int(h["y"])
        hole_list.append((hid, x, y))

    # ✅ id 기준 정렬 (원하면 삭제 가능)
    hole_list.sort(key=lambda t: t[0])

    return hole_list, radius


def detect_table_holes(frame, debug: bool = False):
    holes, r = load_baseline_holes()

    centers = [(x, y) for (_, x, y) in holes]
    radii = [r for _ in centers]

    debug_img = frame.copy()
    if debug:
        for hid, x, y in holes:
            cv2.circle(debug_img, (x, y), r, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.circle(debug_img, (x, y), 2, (0, 0, 255), 3, cv2.LINE_AA)

            # ✅ enumerate 번호 대신 JSON의 id 사용
            label = str(hid)

            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            bx1, by1 = x + r + 2, y - th // 2 - 6
            bx2, by2 = bx1 + tw + 8, by1 + th + 10
            cv2.rectangle(debug_img, (bx1, by1), (bx2, by2), (0, 0, 0), -1)
            cv2.putText(
                debug_img,
                label,
                (bx1 + 4, by1 + th + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        # cv2.putText(
        #     debug_img,
        #     f"holes: {len(holes)}",
        #     (10, 30),
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     1.0,
        #     (0, 255, 0),
        #     2,
        #     cv2.LINE_AA,
        # )

    return centers, radii, debug_img
