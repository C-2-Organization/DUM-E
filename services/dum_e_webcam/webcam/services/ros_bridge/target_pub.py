# webcam/services/ros_bridge/target_pub.py
from __future__ import annotations

import json
import threading
import time
from typing import Any, Dict, Optional

import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
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
    FastAPI/worker에서 안전하게 쓰는 ROS2 Publisher 래퍼.
    - Executor를 별도 스레드에서 spin 해서 DDS 송신 안정화
    - ready 이벤트로 초기 유실 방지
    """
    def __init__(self, topic: str = "/dum_e/webcam/webcam"):
        self.topic = topic
        self._node: Optional[PerceptionPublisher] = None
        self._exec: Optional[SingleThreadedExecutor] = None
        self._thread: Optional[threading.Thread] = None
        self._ready = threading.Event()
        self._stop = threading.Event()
        self._init_lock = threading.Lock()

        # (너무 조용히 버려져서 디버깅 힘드니까) 드롭 로그용
        self._last_drop_log = 0.0

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return

        self._ready.clear()
        self._stop.clear()

        def _spin():
            # rclpy.init은 프로세스당 1회만
            with self._init_lock:
                if not rclpy.ok():
                    rclpy.init(args=None)

            self._node = PerceptionPublisher(topic=self.topic)
            self._exec = SingleThreadedExecutor()
            self._exec.add_node(self._node)
            self._ready.set()

            try:
                while rclpy.ok() and not self._stop.is_set():
                    # spin_once로 돌리면 FastAPI/다른 스레드랑 공존 잘 됨
                    self._exec.spin_once(timeout_sec=0.1)
            finally:
                try:
                    if self._exec and self._node:
                        self._exec.remove_node(self._node)
                except Exception:
                    pass
                try:
                    if self._node:
                        self._node.destroy_node()
                except Exception:
                    pass
                try:
                    if rclpy.ok():
                        rclpy.shutdown()
                except Exception:
                    pass

        self._thread = threading.Thread(target=_spin, daemon=True)
        self._thread.start()

        # ready 대기 (초기 유실 방지)
        self._ready.wait(timeout=2.0)

    def publish(self, data: Dict[str, Any]) -> None:
        if not self._ready.is_set() or self._node is None:
            now = time.time()
            if now - self._last_drop_log > 2.0:
                self._last_drop_log = now
                # 필요하면 여기 print 켜서 "초기화 안 됨"을 확실히 보자
                print("[ROS PUB] drop: publisher not ready")
            return
        self._node.publish_dict(data)

    def stop(self) -> None:
        self._stop.set()
