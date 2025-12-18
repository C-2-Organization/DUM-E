# dum_e_perception/hand_detector.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2

try:
    import mediapipe as mp
except ImportError as e:
    mp = None


@dataclass
class HandDetection:
    u: int
    v: int
    confidence: float
    handedness: str  # "Left" or "Right" or "Unknown"


class MediaPipeHandDetector:
    """
    MediaPipe Hands wrapper.
    - input: BGR image (OpenCV)
    - output: HandDetection (pixel u,v + confidence)
    """

    def __init__(
        self,
        *,
        max_num_hands: int = 1,
        min_detection_confidence: float = 0.35,
        min_tracking_confidence: float = 0.35,
    ):
        if mp is None:
            raise RuntimeError(
                "mediapipe is not installed. Install with: pip install mediapipe"
            )

        self._mp_hands = mp.solutions.hands
        self._hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            model_complexity=1,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def detect(self, color_bgr, mode: str = "palm_center") -> Optional[HandDetection]:
        """
        mode:
          - "palm_center": 손바닥 중심 (기존 handover 용)
          - "index_tip":   검지손가락 끝 (PLACEMP 용)
        """
        if color_bgr is None:
            return None

        h, w = color_bgr.shape[:2]
        if h <= 0 or w <= 0:
            return None

        # MediaPipe는 RGB
        color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
        res = self._hands.process(color_rgb)

        if not res.multi_hand_landmarks:
            return None

        # best hand = handedness score가 있으면 그걸, 없으면 첫 손
        best_idx = 0
        best_score = 0.0
        best_label = "Unknown"

        if res.multi_handedness:
            for i, hd in enumerate(res.multi_handedness):
                try:
                    score = float(hd.classification[0].score)
                    label = str(hd.classification[0].label)
                except Exception:
                    score = 0.0
                    label = "Unknown"
                if score > best_score:
                    best_score = score
                    best_idx = i
                    best_label = label

        lm = res.multi_hand_landmarks[best_idx].landmark

        # 📌 모드에 따라 좌표 선택
        if mode == "index_tip":
            # MediaPipe Hands에서 검지손가락 tip = 8번 landmark
            cx = lm[8].x
            cy = lm[8].y
        else:
            # 기본: 손바닥 중심 (wrist + MCP 평균)
            idxs = [0, 5, 9, 13, 17]
            xs = [lm[i].x for i in idxs]
            ys = [lm[i].y for i in idxs]
            cx = sum(xs) / len(xs)
            cy = sum(ys) / len(ys)

        u = int(round(cx * w))
        v = int(round(cy * h))
        u = max(0, min(w - 1, u))
        v = max(0, min(h - 1, v))

        conf = best_score if best_score > 0.0 else 0.5  # handedness 없을 때 기본값
        return HandDetection(u=u, v=v, confidence=conf, handedness=best_label)
