# webcam/services/yolo_locator.py
from typing import List, Dict, Optional
import numpy as np

from ultralytics import YOLOWorld  # ultralytics==8.3.235 기준

class YoloLocator:
    def __init__(self, model_path: str, conf: float = 0.35, imgsz: int = 640, device: str = "cpu"):
        self.model = YOLOWorld(model_path)
        self.conf = conf
        self.imgsz = imgsz
        self.device = device

    def detect(self, frame_bgr) -> List[Dict]:
        """
        return: [
          {
            "conf": 0.3,
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

        # 신뢰도 높은 순 정렬
        out.sort(key=lambda d: d["conf"], reverse=True)
        return out
