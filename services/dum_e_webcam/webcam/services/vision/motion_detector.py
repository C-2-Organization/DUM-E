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
    # webcam/services/motion_detector.py (추가)

    def get_motion_bbox(self, frame):
        """
        모션이 잡힌 영역의 bbox + 중심점 반환.
        return: (suspicious:bool, bbox:(x,y,w,h)|None, center:(cx,cy)|None)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.prev_gray is None:
            self.prev_gray = gray
            return False, None, None

        diff = cv2.absdiff(self.prev_gray, gray)
        self.prev_gray = gray

        _, thresh = cv2.threshold(diff, self.threshold, 255, cv2.THRESH_BINARY)

        # 노이즈 정리
        thresh = cv2.medianBlur(thresh, 5)
        thresh = cv2.dilate(thresh, None, iterations=2)

        motion_score = np.sum(thresh) / 255.0
        suspicious = motion_score > self.motion_limit
        if not suspicious:
            return False, None, None

        # 큰 덩어리 하나 bbox
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return True, None, None

        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        cx, cy = x + w // 2, y + h // 2
        return True, (x, y, w, h), (cx, cy)
