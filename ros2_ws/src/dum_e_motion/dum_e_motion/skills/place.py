# dum_e_motion/skills/place.py
import json
from typing import Tuple

from geometry_msgs.msg import PoseStamped

from dum_e_interfaces.msg import SkillCommand
from dum_e_motion.motion_context import MotionContext

PLACE_MIN_Z_M = -10.0
PLACE_MAX_Z_M =  10.0

# ------------------------------------------------------------------
# Doosan + RG2 place 모션
# ------------------------------------------------------------------
def execute_place_motion(ctx: MotionContext, x, y, z):
    """
    목표 좌표로 이동 → 그리퍼 오픈(드롭) → 홈
    """
    from DSR_ROBOT2 import (
        DR_MV_MOD_ABS,
        DR_MV_RA_DUPLICATE,
        get_current_posx,
    )
    from DR_common2 import posx

    ctx.node.get_logger().info(
        f"[MOVE] Place → base({x:.3f}, {y:.3f}, {z:.3f})"
    )

    current_pos = get_current_posx()[0]

    target_pos = posx([
        x,
        y,
        z,
        current_pos[3],
        current_pos[4],
        current_pos[5],
    ])

    # 1) 목표 지점으로 이동
    ctx.motion.movel(
        target_pos,
        vel=ctx.LIN_VEL,
        acc=ctx.LIN_ACC,
        mod=DR_MV_MOD_ABS,
        ra=DR_MV_RA_DUPLICATE,
    )

    # 2) 놓기 (그리퍼 오픈)
    ctx.motion.open_gripper()
    ctx.motion.wait(1)

    # 3) 홈으로 복귀
    ctx.motion.movej(
        ctx.CUSTOM_HOME_JOINT,
        vel=ctx.JNT_VEL,
        acc=ctx.JNT_ACC,
        mod=DR_MV_MOD_ABS,
        ra=DR_MV_RA_DUPLICATE,
    )


# ------------------------------------------------------------------
# Place service call example
# ------------------------------------------------------------------
# ros2 service call /run_skill dum_e_interfaces/srv/RunSkill "{
#   command: {
#     skill_type: 4,
#     object_name: 'shelf'
#     params_json: ''
#   }
# }"

def run_place_skill(
    cmd: SkillCommand,
    ctx: MotionContext,
) -> Tuple[bool, str, float, PoseStamped]:
    """
    PLACE 스킬 실행:
      - cmd.target_pose 사용 (frame_id는 'base' 권장)
      - target_pose 좌표로 이동 후 open_gripper
      - (success, message, confidence, final_pose) 반환
    """
    location = cmd.object_name.strip()
    params_json = cmd.params_json

    if not location:
        msg = "location is empty"
        ctx.node.get_logger().warn(f"[PLACE] {msg}")
        return False, msg, 0.0, PoseStamped()

    ctx.node.get_logger().info(
        f"[PLACE] skill 실행: location='{location}'"
    )

    # params_json 파싱 (추후: approach_offset, retract, open_wait 등 옵션 확장용)
    if params_json:
        try:
            params = json.loads(params_json)
            ctx.node.get_logger().info(f"[PLACE] params_json = {params}")
        except json.JSONDecodeError:
            ctx.node.get_logger().warn(f"[PLACE] params_json parse failed: {params_json}")

    # TODO: Replace to actual location
    x = 368.0
    y = 21.0
    z = 350.0

    try:
        execute_place_motion(ctx, x, y, z)
        success = True
        message = "OK"
        confidence = 1.0
    except Exception as e:
        success = False
        message = f"place motion error: {e}"
        confidence = 0.0
        ctx.node.get_logger().error(f"❌ place motion 중 예외: {e}")

    final_pose = ctx.make_final_pose(x, y, z)
    return success, message, confidence, final_pose
