# dum_e_motion/skills/drop.py
from typing import Tuple
from dum_e_motion.motion_context import MotionContext

OPEN_EPS_MM = 5.0

def _is_gripper_open(ctx: MotionContext) -> bool:
    width_mm = ctx.gripper.get_width()
    max_width_mm = ctx.gripper.max_width / 10.0
    is_open = width_mm >= (max_width_mm - OPEN_EPS_MM)

    return is_open

def execute_drop_motion(ctx: MotionContext):
    if _is_gripper_open(ctx):
        raise RuntimeError("Nothing to drop: gripper is already open and no grip is detected.")
    ctx.motion.open_gripper()
    ctx.motion.wait(1)

# ------------------------------------------------------------------
# Drop service call example
# ------------------------------------------------------------------
# ros2 service call /run_skill dum_e_interfaces/srv/RunSkill "{ command: { skill_type: 3 } }"

def run_drop_skill(ctx: MotionContext) -> Tuple[bool, str, float]:
    try:
        execute_drop_motion(ctx)
        success = True
        message = "OK"
        confidence = 1.0
    except Exception as e:
        success = False
        message = f"drop motion error: {e}"
        confidence = 0.0
        ctx.node.get_logger.error(f"❌ drop motion 중 예외: {e}")
    return success, message, confidence
