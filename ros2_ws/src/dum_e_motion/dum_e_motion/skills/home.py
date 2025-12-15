# dum_e_motion/skills/home.py
from typing import Tuple
from geometry_msgs.msg import PoseStamped
from dum_e_interfaces.msg import SkillCommand
from dum_e_motion.motion_context import MotionContext

def execute_home_motion(ctx: MotionContext):
    from DSR_ROBOT2 import (
        DR_MV_MOD_ABS,
        DR_MV_RA_DUPLICATE,
    )
    ctx.motion.open_gripper()

    ctx.motion.wait(1)

    ctx.motion.movej(
        ctx.CUSTOM_HOME_JOINT,
        vel=ctx.JNT_VEL,
        acc=ctx.JNT_ACC,
        mod=DR_MV_MOD_ABS,
        ra=DR_MV_RA_DUPLICATE,
    )

# ------------------------------------------------------------------
# Home service call example
# ------------------------------------------------------------------
# ros2 service call /run_skill dum_e_interfaces/srv/RunSkill "{ command: { skill_type: 2 } }"

def run_home_skill(ctx: MotionContext) -> Tuple[bool, str, float, PoseStamped]:
    from DSR_ROBOT2 import(
        get_current_posx
    )
    try:
        execute_home_motion(ctx)
        success = True
        message = "OK"
        confidence = 1.0
    except Exception as e:
        success = False
        message = f"home motion error: {e}"
        confidence = 0.0
        ctx.node.get_logger.error(f"❌ home motion 중 예외: {e}")
    current_pos = get_current_posx()[0]
    final_pose = ctx.make_final_pose(current_pos[0], current_pos[1], current_pos[2])
    return success, message, confidence, final_pose
