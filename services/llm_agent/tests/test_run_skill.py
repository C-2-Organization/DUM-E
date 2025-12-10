# tests/test_run_skill.py
from __future__ import annotations

import os
import sys

# === 상위 디렉토리(services/llm_agent)를 모듈 검색 경로에 추가 ===
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from geometry_msgs.msg import PoseStamped
from dum_e_interfaces.msg import SkillCommand

from ros_bridge import call_run_skill, shutdown_ros  # 이제 잘 불러와질 것


def build_test_pose() -> PoseStamped:
    pose = PoseStamped()
    pose.header.frame_id = "base_link"
    pose.pose.position.x = 0.0
    pose.pose.position.y = 0.0
    pose.pose.position.z = 0.0
    pose.pose.orientation.w = 1.0
    return pose


def main():
    """
    dum_e_bringup 이 떠 있는 상태에서 테스트하는 스크립트.

    1. /run_skill 을 PICK + scissors 로 호출
    2. 응답 success / confidence / message 출력
    3. 간단한 assert 로 형식 검증
    """

    print("[TEST] calling /run_skill with PICK + 'scissors' ...")

    pose = build_test_pose()

    resp = call_run_skill(
        skill_type=SkillCommand.PICK,
        object_name="scissors",
        target_pose=None,
        params_json="",
        timeout_sec=20.0,
    )

    print(f"[TEST] success   : {resp.success}")
    print(f"[TEST] message   : {resp.message}")
    print(f"[TEST] confidence: {resp.confidence}")
    print(f"[TEST] final_pose frame_id: {resp.final_pose.header.frame_id}")
    print(
        f"[TEST] final_pose position: "
        f"({resp.final_pose.pose.position.x}, "
        f"{resp.final_pose.pose.position.y}, "
        f"{resp.final_pose.pose.position.z})"
    )

    # --- 간단한 검증 ---
    assert isinstance(resp.success, bool), "success should be bool"
    assert isinstance(resp.message, str), "message should be string"
    if resp.success:
        assert 0.0 <= resp.confidence <= 1.0, "confidence must be in [0, 1]"

    print("[TEST] /run_skill call test passed.")

    shutdown_ros()


if __name__ == "__main__":
    main()
