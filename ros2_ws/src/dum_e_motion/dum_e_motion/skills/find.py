# dum_e_motion/skills/find.py

import json
import time
from typing import Tuple

from geometry_msgs.msg import PoseStamped

from dum_e_interfaces.msg import SkillCommand
from dum_e_motion.motion_context import MotionContext

# 이 이상일 때 "찾았다"라고 인정
FIND_CONF_TH = 0.3

# 스캔 패턴 설정 (deg)
SCAN_STEP_DEG = 15.0
SCAN_MAX_DEG = 60.0

JNT_VEL = 20.0
JNT_ACC = 120.0

def execute_find_scan_step(ctx: MotionContext, step_idx: int):
    """
    아주 단순한 스윕 패턴:

      - base joint(j1)를 좌/우로 번갈아가며 회전
      - step_idx:
          0 -> +SCAN_STEP_DEG
          1 -> -SCAN_STEP_DEG
          2 -> +2*SCAN_STEP_DEG
          3 -> -2*SCAN_STEP_DEG
          ...
      - 최대 각도는 ±SCAN_MAX_DEG로 클램프
    """
    from DSR_ROBOT2 import (
        movej,
        posj,
        DR_MV_MOD_REL,
        DR_MV_RA_DUPLICATE,
    )

    k = step_idx // 2 + 1
    sign = 1 if (step_idx % 2 == 0) else -1  # 짝수: +, 홀수: -

    delta_x = sign * k * SCAN_STEP_DEG
    delta_y =  (-0.5) * SCAN_STEP_DEG

    target_j = posj(delta_x, 0, 0, 0, delta_y)

    movej(
        target_j,
        vel=JNT_VEL,
        acc=JNT_ACC,
        mod=DR_MV_MOD_REL,
        ra=DR_MV_RA_DUPLICATE,
    )

# ------------------------------------------------------------------
# Find service call example
# ------------------------------------------------------------------
# ros2 service call /run_skill dum_e_interfaces/srv/RunSkill "{
#   command: {
#     skill_type: 1,
#     object_name: 'scissors',
#     target_pose: {
#       header: {frame_id: ''},
#       pose: {
#         position: {x: 0.0, y: 0.0, z: 0.0},
#         orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}
#       }
#     },
#     params_json: '{\"max_search_time\": 20.0, \"scan_interval\": 0.5}'
#   }
# }"

def run_find_skill(
    cmd: SkillCommand,
    ctx: MotionContext,
) -> Tuple[bool, str, float, PoseStamped]:
    """
    FIND 스킬:

      - object_name에 대해 get_object_pose()를 반복 호출
      - 디텍션 안 되면 스윕 모션(예: base joint 좌우 회전) 수행
      - conf >= FIND_CONF_TH 인 디텍션이 나오면 그 즉시 성공 반환
      - 포즈는 여기서 안 쓰고, 다음 스킬(Pick 등)에서 다시 perception 호출해서 쓰도록 설계

    params_json 예시:
      {
        "max_search_time": 10.0,   # seconds (default 10)
        "scan_interval": 1.0       # seconds, 모션 스텝 간격 (default 1)
      }

    return:
      success, message, confidence, final_pose
      - final_pose: Find는 "자세 찾기" 용이라 여기서는 빈 PoseStamped() 반환
    """
    from DSR_ROBOT2 import (
        movej,
        posj,
        DR_MV_MOD_ABS,
        DR_MV_RA_DUPLICATE,
    )

    object_name = cmd.object_name.strip()
    params_json = cmd.params_json

    if not object_name:
        msg = "object_name is empty"
        ctx.node.get_logger().warn(f"[FIND] {msg}")
        return False, msg, 0.0, PoseStamped()

    # 기본 파라미터
    max_search_time = 10.0
    scan_interval = 1.0

    # params_json 파싱
    if params_json:
        try:
            params = json.loads(params_json)
            max_search_time = float(params.get("max_search_time", max_search_time))
            scan_interval = float(params.get("scan_interval", scan_interval))
            ctx.node.get_logger().info(
                f"[FIND] params: max_search_time={max_search_time}, "
                f"scan_interval={scan_interval}"
            )
        except json.JSONDecodeError:
            ctx.node.get_logger().warn(
                f"[FIND] params_json 파싱 실패: {params_json}"
            )

    ctx.node.get_logger().info(
        f"[FIND] skill 실행: object_name='{object_name}', "
        f"max_search_time={max_search_time}s"
    )

    start_time = time.time()
    last_scan_time = start_time - scan_interval  # 시작하자마자 한 번 움직이게
    step_idx = 0

    START_POSE = posj(0.0, 0.0, 90.0, 0.0, 120.0, 90.0)

    movej(
        START_POSE,
        vel=JNT_VEL,
        acc=JNT_ACC,
        mod=DR_MV_MOD_ABS,
        ra=DR_MV_RA_DUPLICATE
    )

    while True:
        now = time.time()
        elapsed = now - start_time

        # 1) 타임아웃
        if elapsed > max_search_time:
            msg = (
                f"[FIND] timeout: {elapsed:.1f}s > max_search_time={max_search_time:.1f}s, "
                "object not found"
            )
            ctx.node.get_logger().warn(msg)
            return False, msg, 0.0, PoseStamped()

        # 2) 현재 자세에서 object pose 시도
        pose_resp = ctx.request_object_pose(object_name)
        if pose_resp is not None and pose_resp.success:
            conf = float(pose_resp.confidence)
            if conf >= FIND_CONF_TH:
                ctx.node.get_logger().info(
                    f"[FIND] object detected: conf={conf:.2f} >= {FIND_CONF_TH:.2f}, 탐색 종료"
                )
                # 여기서는 순수하게 "찾았다"만 알리고 포즈는 안 넘김
                msg = "object found"
                return True, msg, conf, PoseStamped()
            else:
                ctx.node.get_logger().info(
                    f"[FIND] detected but conf={conf:.2f} < TH={FIND_CONF_TH:.2f}, 계속 탐색"
                )

        # 3) 스윕 모션: 일정 시간마다 한 스텝씩 자세 변경
        if now - last_scan_time >= scan_interval:
            ctx.node.get_logger().info(
                f"[FIND] scan step {step_idx} (elapsed={elapsed:.1f}s)"
            )
            try:
                execute_find_scan_step(ctx, step_idx)
            except Exception as e:
                ctx.node.get_logger().error(f"[FIND] scan step error: {e}")
            step_idx += 1
            last_scan_time = now

        # 4) 너무 바쁘게 돌지 않게 조금 쉼
        time.sleep(0.1)
