# dum_e_perception/remote_detector.py
from __future__ import annotations

import cv2
import requests


class RemoteVisionDetector:
    """
    Remote detector (GroundingDINO server) -> returns detections with normalized bbox.
    Output format matches YOLODetector.detect():
      [
        {
          "class_name": str,
          "confidence": float,
          "bbox": [x1, y1, x2, y2]  # normalized 0~1
          "source": "remote_gdino"
        }
      ]
    """

    def __init__(self, endpoint: str, timeout_sec: float = 20.0):
        self.endpoint = endpoint.rstrip("/")
        self.timeout_sec = timeout_sec

    def detect(
        self,
        image_bgr,
        text_prompt: str,
        top_k: int = 5,
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
    ):
        h, w = image_bgr.shape[:2]

        ok, jpg = cv2.imencode(".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if not ok:
            return []

        files = {"image": ("frame.jpg", jpg.tobytes(), "image/jpeg")}
        data = {
            "text_prompt": text_prompt,
            "top_k": str(top_k),
            "box_threshold": str(box_threshold),
            "text_threshold": str(text_threshold),
        }

        r = requests.post(self.endpoint, files=files, data=data, timeout=self.timeout_sec)
        r.raise_for_status()
        js = r.json()

        out = []
        for d in js.get("detections", []):
            bbox_xyxy = d.get("bbox_xyxy", None)
            if not bbox_xyxy or len(bbox_xyxy) != 4:
                continue

            x1, y1, x2, y2 = bbox_xyxy  # pixels
            # clamp to image bounds
            x1 = max(0.0, min(float(w - 1), float(x1)))
            y1 = max(0.0, min(float(h - 1), float(y1)))
            x2 = max(0.0, min(float(w - 1), float(x2)))
            y2 = max(0.0, min(float(h - 1), float(y2)))
            if x2 <= x1 or y2 <= y1:
                continue

            # convert to normalized bbox (0~1)
            out.append({
                "bbox": [x1 / w, y1 / h, x2 / w, y2 / h],
                "confidence": float(d.get("score", 0.0)),
                "class_name": str(d.get("phrase", text_prompt)),
                "source": "remote_gdino",
            })

        return out
