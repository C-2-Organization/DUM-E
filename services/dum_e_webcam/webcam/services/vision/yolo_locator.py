# webcam/services/yolo_locator.py
from typing import List, Dict, Optional, Tuple
import math

from ultralytics import YOLOWorld  # ultralytics==8.3.235 기준


def _dist(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


class YoloLocator:
    def __init__(
        self,
        model_path: str,
        conf: float = 0.05,
        imgsz: int = 1280,
        device: str = "cpu",
        # ✅ 안정화(트래킹) 파라미터
        match_dist: int = 60,   # 중심점 거리 이내면 같은 물체로 간주
        confirm_hits: int = 3,  # 3프레임 연속(누적) 보이면 확정
        max_miss: int = 5,      # 몇 프레임까지 안 보이면 트랙 삭제
        match_by_class: bool = True,  # cls_name 같은 것끼리만 매칭
    ):
        self.model = YOLOWorld(model_path)
        
        self.model.set_classes([
            "person", "hand",
            "scissors", "knife", "box cutter", "hammer",
            "phone",
            "tool", "object"
        ])
        
        self.conf = conf
        self.imgsz = imgsz
        self.device = device

        self.match_dist = match_dist
        self.confirm_hits = confirm_hits
        self.max_miss = max_miss
        self.match_by_class = match_by_class

        # ✅ 트랙 상태 유지
        self._tracks: List[Dict] = []
        self._next_id: int = 1

    def reset_tracks(self):
        """상황 초기화(카메라 재시작 등) 시 호출"""
        self._tracks.clear()
        self._next_id = 1

    def detect(self, frame_bgr) -> List[Dict]:
        """
        후보(det)만 반환 (원본과 동일)
        return: [
          {
            "cls_name": str,
            "conf": float,
            "bbox": (x1,y1,x2,y2),
            "center": (cx,cy)
          }, ...
        ]
        """
        results = self.model.predict(
            source=frame_bgr,
            conf=self.conf,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False
        )
        
        out: List[Dict] = []
        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return out

        names = r.names  # class id -> name
        boxes = r.boxes

        xyxy = boxes.xyxy.cpu().numpy()  # (N,4)
        confs = boxes.conf.cpu().numpy() # (N,)
        clss = boxes.cls.cpu().numpy()   # (N,)

        for (x1, y1, x2, y2), cf, c in zip(xyxy, confs, clss):
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            cls_id = int(c)
            cls_name = names.get(cls_id, str(cls_id))

            out.append({
                "cls_name": cls_name,
                "conf": float(cf),
                "bbox": (int(x1), int(y1), int(x2), int(y2)),
                "center": (cx, cy),
            })
            
        # print("YOLO dets:", len(out), "top_conf:", (out[0]["conf"] if out else None))


        out.sort(key=lambda d: d["conf"], reverse=True)
        return out

    def detect_confirmed(self, frame_bgr) -> List[Dict]:
        """
        ✅ 후보를 detect한 다음,
        ✅ 3프레임 안정화된 것만 confirmed로 반환
        return: [
          {
            "track_id": 1,
            "cls_name": "...",
            "conf": 0.32,
            "bbox": (...),
            "center": (...),
            "hit": 5,
            "miss": 0,
          }, ...
        ]
        """
        dets = self.detect(frame_bgr)
        self._update_tracks(dets)

        confirmed = [
            {
                "track_id": tr["id"],
                "cls_name": tr["cls_name"],
                "conf": tr["conf"],
                "bbox": tr["bbox"],
                "center": tr["center"],
                "hit": tr["hit"],
                "miss": tr["miss"],
            }
            for tr in self._tracks
            if tr["hit"] >= self.confirm_hits and tr["miss"] == 0  # 지금 보이는 것만
        ]

        # hit 높은 순으로 정렬
        confirmed.sort(key=lambda d: (d["hit"], d["conf"]), reverse=True)
        return confirmed

    def _update_tracks(self, dets: List[Dict]) -> None:
        used = [False] * len(dets)

        # 1) 기존 트랙에 매칭
        for tr in self._tracks:
            best_i = -1
            best_d = 1e9

            for i, d in enumerate(dets):
                if used[i]:
                    continue
                if self.match_by_class and d["cls_name"] != tr["cls_name"]:
                    continue

                dd = _dist(d["center"], tr["center"])
                if dd < best_d:
                    best_d = dd
                    best_i = i

            if best_i != -1 and best_d <= self.match_dist:
                d = dets[best_i]
                used[best_i] = True
                tr["center"] = d["center"]
                tr["bbox"] = d["bbox"]
                tr["conf"] = d["conf"]
                tr["cls_name"] = d["cls_name"]
                tr["hit"] += 1
                tr["miss"] = 0
            else:
                tr["miss"] += 1

        # 2) 새 트랙 생성
        for i, d in enumerate(dets):
            if used[i]:
                continue
            self._tracks.append({
                "id": self._next_id,
                "cls_name": d["cls_name"],
                "conf": d["conf"],
                "bbox": d["bbox"],
                "center": d["center"],
                "hit": 1,
                "miss": 0,
            })
            self._next_id += 1

        # 3) 오래 안 보인 트랙 삭제
        self._tracks = [t for t in self._tracks if t["miss"] <= self.max_miss]
