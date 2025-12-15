# webcam/services/ros_bridge/target_pub.py
from __future__ import annotations
import json
import threading
from typing import Any, Dict, Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class PerceptionPublisher(Node):
    def __init__(self, topic: str = "/dum_e/webcam/webcam"):
        super().__init__("webcam_webcam_pub")
        self.pub = self.create_publisher(String, topic, 10)

    def publish_dict(self, data: Dict[str, Any]) -> None:
        msg = String()
        msg.data = json.dumps(data, ensure_ascii=False)
        self.pub.publish(msg)


class PerceptionPublisherThread:
    """
    FastAPI/worker 코드에서 쉽게 쓰려고 스레드 래퍼 제공
    """
    def __init__(self, topic: str = "/dum_e/webcam/webcam"):
        self.topic = topic
        self._node: Optional[PerceptionPublisher] = None
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return

        def _spin():
            rclpy.init(args=None)
            self._node = PerceptionPublisher(topic=self.topic)
            rclpy.spin(self._node)
            self._node.destroy_node()
            rclpy.shutdown()

        self._thread = threading.Thread(target=_spin, daemon=True)
        self._thread.start()

    def publish(self, data: Dict[str, Any]) -> None:
        if self._node is None:
            return
        self._node.publish_dict(data)
