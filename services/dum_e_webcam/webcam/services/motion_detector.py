# webcam/services/motion_detector.py
import cv2
import numpy as np


class MotionDetector:
    """아주 단순한 프레임 차이 기반 모션 디텍터"""

    def __init__(self, threshold: int = 25, motion_limit: float = 5000.0):
        self.prev_gray = None
        self.threshold = threshold
        self.motion_limit = motion_limit

    def is_suspicious(self, frame) -> bool:
        """이전 프레임과 비교해서 모션이 일정 이상이면 True"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.prev_gray is None:
            self.prev_gray = gray
            return False

        diff = cv2.absdiff(self.prev_gray, gray)
        self.prev_gray = gray

        _, thresh = cv2.threshold(
            diff, self.threshold, 255, cv2.THRESH_BINARY
        )
        motion_score = np.sum(thresh) / 255.0

        return motion_score > self.motion_limit
