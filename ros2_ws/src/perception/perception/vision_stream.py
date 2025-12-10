#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

import cv2
from ultralytics import YOLOWorld

from utils.realsense import ImgNode

# TODO: Get path from env
MODEL_PATH = "/home/rokey/DUM-E/models/yolov8s-worldv2.pt"
YOLO_CLASSES = [
    "a person",
    "a coffee mug",
    "a pair of scissors",
    "a utility knife",
    "a hammer",
    "a screwdriver",
    "a laptop computer",
]

YOLO_CONF_TH = 0.25


class VisionStream(Node):
    def __init__(self):
        super().__init__("vision_debug_node")

        # RealSense 구독 노드
        self.img_node = ImgNode()

        # YOLOWorld 모델 로드
        self.yolo_model = YOLOWorld(MODEL_PATH)
        self.yolo_model.set_classes(YOLO_CLASSES)

        self.get_logger().info("✅ VisionStream started. Press ESC to quit.")

    def run(self):
        cv2.namedWindow("VisionDebug", cv2.WINDOW_NORMAL)

        while rclpy.ok():
            # RealSense 새 프레임 수신
            rclpy.spin_once(self.img_node, timeout_sec=0.1)
            color_img = self.img_node.get_color_frame()
            if color_img is None:
                continue

            # YOLO 추론
            results = self.yolo_model.predict(
                source=color_img,
                conf=YOLO_CONF_TH,
                imgsz=640,
                verbose=False,
            )
            res = results[0]
            annotated = res.plot()

            cv2.imshow("VisionDebug", annotated)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break

        cv2.destroyAllWindows()
        self.get_logger().info("👋 VisionStream finished.")


def main(args=None):
    rclpy.init(args=args)

    node = VisionStream()
    try:
        node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
