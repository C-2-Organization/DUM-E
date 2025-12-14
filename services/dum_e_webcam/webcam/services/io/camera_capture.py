# webcam/services/camera_capture.py
import cv2


class CameraCapture:
    def __init__(self, cam_id: int = 0):
        self.cap = cv2.VideoCapture(cam_id)

    def read(self):
        """한 프레임 읽어서 반환, 실패 시 None"""
        if not self.cap.isOpened():
            return None
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    def release(self):
        if self.cap.isOpened():
            self.cap.release()
