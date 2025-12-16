# dum_e_motion/skills/look_at.py
import json
from typing import Tuple
from geometry_msgs.msg import PoseStamped
from dum_e_interfaces.msg import SkillCommand
from dum_e_motion.motion_context import MotionContext


def _choose_best_candidate(candidates):
    # 1) in_table_roi True 우선
    in_roi = [c for c in candidates if c.get("in_table_roi") is True]
    pool = in_roi if in_roi else candidates

    # 2) hit 큰 순, conf 큰 순
    pool.sort(key=lambda d: (float(d.get("hit", 0)), float(d.get("conf", 0.0))), reverse=True)
    return pool[0] if pool else None


def execute_look_at(ctx: MotionContext, x_mm: float, y_mm: float, z_mm: float):
    from DSR_ROBOT2 import DR_MV_MOD_ABS, DR_MV_RA_DUPLICATE, get_current_posx
    from DR_common2 import posx

    cur = get_current_posx()[0]
    rx, ry, rz = cur[3], cur[4], cur[5]

    ctx.node.get_logger().info(
        f"[LOOK_AT] movel base(mm)=({x_mm:.1f},{y_mm:.1f},{z_mm:.1f}) "
        f"r(cur)=({rx:.1f},{ry:.1f},{rz:.1f})"
    )

    ctx.motion.movel(
        posx([x_mm, y_mm, z_mm, rx, ry, rz]),
        vel=ctx.LIN_VEL,
        acc=ctx.LIN_ACC,
        mod=DR_MV_MOD_ABS,
        ra=DR_MV_RA_DUPLICATE,
    )

    ctx.motion.wait(0.2)

def run_look_at_skill(cmd: SkillCommand, ctx: MotionContext) -> Tuple[bool, str, float, PoseStamped]:
    """
    params_json 기대:
    {
      "candidates": [
        {"robot_xy":[x_mm,y_mm], "conf":0.3, "hit":3, "in_table_roi":true, ...},
        ...
      ],
      "z_mm": 500.0,
      "offset_x": 0.0, "offset_y": 0.0
    }

    - rx/ry/rz는 받더라도 사용하지 않습니다(현재 자세 유지).
    """
    if not cmd.params_json:
        return False, "params_json empty", 0.0, PoseStamped()

    try:
        params = json.loads(cmd.params_json)
    except Exception as e:
        return False, f"params_json parse fail: {e}", 0.0, PoseStamped()

    candidates = params.get("candidates") or []
    if not candidates:
        return False, "no candidates", 0.0, PoseStamped()

    best = _choose_best_candidate(candidates)
    if not best:
        return False, "best candidate not found", 0.0, PoseStamped()

    robot_xy = best.get("robot_xy")
    if not (isinstance(robot_xy, (list, tuple)) and len(robot_xy) == 2):
        return False, "best candidate has no robot_xy", float(best.get("conf", 0.0)), PoseStamped()

    x_mm, y_mm = float(robot_xy[0]), float(robot_xy[1])

    # ✅ 헴: look_at은 z 고정, r은 current_posx 사용
    z_mm = 200.0
    offset_x = float(params.get("offset_x", 0.0))
    offset_y = float(params.get("offset_y", 0.0))

    x_mm += offset_x
    y_mm += offset_y

    try:
        execute_look_at(ctx, x_mm, y_mm, z_mm)
        return True, "OK", float(best.get("conf", 1.0)), PoseStamped()
    except Exception as e:
        return False, f"look_at motion error: {e}", float(best.get("conf", 0.0)), PoseStamped()
