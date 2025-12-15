# dum_e_perception_bridge/receiver.py
import json
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from .lookat_map import get_lookat_pose

class WebcamReceiver(Node):
    def __init__(self):
        super().__init__("webcam_receiver")
        self.create_subscription(String, "/dum_e/perception/webcam", self.cb, 10)

    def cb(self, msg: String):
        try:
            data = json.loads(msg.data)
        except Exception as e:
            self.get_logger().warn(f"JSON parse fail: {e}")
            return

        action = data.get("recommended_action")
        cands = data.get("candidates") or []
        if not cands:
            return

        best = cands[0]
        between = best.get("between_holes")
        pose = get_lookat_pose(between)

        self.get_logger().info(f"action={action} between={between} pose={pose}")

        # TODO: 여기서 실제 로봇 look_at/이동 실행
        # 예: call 서비스 / action client / Doosan API wrapper 호출

def main():
    rclpy.init()
    node = WebcamReceiver()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
