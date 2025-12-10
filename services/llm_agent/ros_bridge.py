# ros_bridge.py
from __future__ import annotations

import threading
from typing import Optional

import rclpy
from rclpy.node import Node

from dum_e_interfaces.srv import RunSkill
from dum_e_interfaces.msg import SkillCommand
from geometry_msgs.msg import PoseStamped


# ---- 전역 상태 (간단한 싱글톤 패턴) ----
_rclpy_initialized = False
_node_lock = threading.Lock()
_node: Optional[Node] = None


def init_ros(node_name: str = "llm_skill_bridge") -> Node:
    """
    rclpy와 노드를 초기화하고 전역 Node를 반환.
    여러 번 호출돼도 한 번만 초기화되도록 보호.
    """
    global _rclpy_initialized, _node

    with _node_lock:
        if not _rclpy_initialized:
            rclpy.init()
            _node = rclpy.create_node(node_name)
            _rclpy_initialized = True

        assert _node is not None
        return _node


def get_node() -> Node:
    """
    이미 초기화된 Node를 가져오거나, 없으면 새로 초기화.
    """
    if _node is None:
        return init_ros()
    return _node


def shutdown_ros():
    """
    테스트 종료 시 깔끔하게 rclpy 종료.
    (FastAPI에서 프로세스가 계속 도는 경우엔 안 써도 됨)
    """
    global _rclpy_initialized, _node
    with _node_lock:
        if _rclpy_initialized:
            if _node is not None:
                _node.destroy_node()
                _node = None
            rclpy.shutdown()
            _rclpy_initialized = False


def _build_default_pose(frame_id: str = "base_link") -> PoseStamped:
    """
    target_pose 를 명시하지 않았을 때 사용할 기본 Pose.
    지금은 0,0,0 + 단위 quaternion 으로 세팅.
    """
    pose = PoseStamped()
    pose.header.frame_id = frame_id
    pose.pose.position.x = 0.0
    pose.pose.position.y = 0.0
    pose.pose.position.z = 0.0
    pose.pose.orientation.x = 0.0
    pose.pose.orientation.y = 0.0
    pose.pose.orientation.z = 0.0
    pose.pose.orientation.w = 1.0
    return pose


def call_run_skill(
    skill_type: int,
    object_name: str = "",
    target_pose: Optional[PoseStamped] = None,
    params_json: str = "",
    timeout_sec: float = 10.0,
) -> RunSkill.Response:
    """
    /run_skill 서비스를 동기적으로 호출하는 헬퍼 함수.

    dum_e_bringup 이 떠 있고, /run_skill 서버가 활성화 되어있다는 전제.
    """

    node = get_node()

    client = node.create_client(RunSkill, "/run_skill")

    # 서비스 서버 대기
    if not client.wait_for_service(timeout_sec=5.0):
        node.get_logger().error("Service /run_skill not available")
        raise RuntimeError("Service /run_skill not available. Is dum_e_bringup running?")

    req = RunSkill.Request()

    # SkillCommand 필드 채우기
    req.command.skill_type = skill_type
    req.command.object_name = object_name

    if target_pose is None:
        req.command.target_pose = PoseStamped()
    else:
        req.command.target_pose = target_pose

    req.command.params_json = params_json or ""

    future = client.call_async(req)

    # future 완료까지 블로킹
    rclpy.spin_until_future_complete(node, future, timeout_sec=timeout_sec)

    if not future.done():
        node.get_logger().error("Timeout while waiting for /run_skill response")
        raise TimeoutError("Timeout while waiting for /run_skill response")

    if future.result() is None:
        raise RuntimeError(f"Service /run_skill call failed: {future.exception()}")

    response: RunSkill.Response = future.result()

    node.get_logger().info(
        f"/run_skill result: success={response.success}, "
        f"confidence={response.confidence:.2f}, message='{response.message}'"
    )

    return response
