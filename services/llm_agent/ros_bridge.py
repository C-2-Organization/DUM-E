from __future__ import annotations

import threading
from typing import Optional

import rclpy
from rclpy.node import Node

from dum_e_interfaces.srv import RunSkill
from geometry_msgs.msg import PoseStamped


# ---- 전역 상태 (간단한 싱글톤 패턴) ----
_rclpy_initialized = False
_node_lock = threading.Lock()
_node: Optional[Node] = None

# ---- 스킬별 기본 timeout 정책 ----
# TRACKING(5)는 "계속 도는" 스킬이라 짧은 timeout이면 무조건 터짐.
_LONG_RUNNING_SKILLS = {5}  # 필요하면 여기에 추가
_DEFAULT_TIMEOUT_SEC = 10.0
_LONG_TIMEOUT_SEC = 300.0  # 


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
    0,0,0 + 단위 quaternion
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


def _choose_timeout(skill_type: int, timeout_sec: Optional[float]) -> float:
    """
    스킬 성격에 따라 timeout 정책을 결정.
    - 호출자가 timeout_sec를 명시하면 그 값을 우선
    - 그렇지 않으면 tracking 같은 장시간 스킬은 LONG timeout
    """
    if timeout_sec is not None:
        return float(timeout_sec)

    if int(skill_type) in _LONG_RUNNING_SKILLS:
        return float(_LONG_TIMEOUT_SEC)

    return float(_DEFAULT_TIMEOUT_SEC)


def call_run_skill(
    skill_type: int,
    object_name: str = "",
    target_pose: Optional[PoseStamped] = None,
    params_json: str = "",
    timeout_sec: Optional[float] = None,
) -> RunSkill.Response:
    """
    /run_skill 서비스를 동기적으로 호출하는 헬퍼 함수.
    - TRACKING(5) 같은 장시간 스킬은 기본 timeout을 크게 설정
    """

    node = get_node()
    effective_timeout = _choose_timeout(skill_type, timeout_sec)

    client = node.create_client(RunSkill, "/run_skill")

    # 서비스 서버 대기
    if not client.wait_for_service(timeout_sec=5.0):
        node.get_logger().error("Service /run_skill not available")
        raise RuntimeError("Service /run_skill not available. Is dum_e_bringup running?")

    req = RunSkill.Request()
    req.command.skill_type = int(skill_type)
    req.command.object_name = object_name

    # ✅ 기존 코드 버그 수정: None이면 빈 PoseStamped()가 아니라 default pose를 넣음
    if target_pose is None:
        req.command.target_pose = _build_default_pose()
    else:
        req.command.target_pose = target_pose

    req.command.params_json = params_json or ""

    future = client.call_async(req)

    # future 완료까지 블로킹 (tracking은 오래 걸릴 수 있으니 timeout 정책 적용)
    rclpy.spin_until_future_complete(node, future, timeout_sec=effective_timeout)

    if not future.done():
        node.get_logger().error(
            f"Timeout while waiting for /run_skill response (skill_type={skill_type}, timeout={effective_timeout}s)"
        )
        # 클라이언트 정리(리소스 누수 방지)
        try:
            node.destroy_client(client)
        except Exception:
            pass
        raise TimeoutError("Timeout while waiting for /run_skill response")

    if future.result() is None:
        ex = future.exception()
        raise RuntimeError(f"Service /run_skill call failed: {ex}")

    response: RunSkill.Response = future.result()

    node.get_logger().info(
        f"/run_skill result: success={response.success}, "
        f"confidence={response.confidence:.2f}, message='{response.message}'"
    )

    # 클라이언트 정리(선택: 반복 호출이 많으면 누적 방지에 도움)
    try:
        node.destroy_client(client)
    except Exception:
        pass

    return response
