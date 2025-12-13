# dum_e_motion/skills/find.py

import json
import time
from typing import Tuple

from geometry_msgs.msg import PoseStamped

from dum_e_interfaces.msg import SkillCommand
from dum_e_motion.motion_context import MotionContext, MotionCancelled

# 이 이상일 때 "찾았다"라고 인정
FIND_CONF_TH = 0.3

# 책상 위(inside) 스캔 패턴 설정 (deg)
SCAN_STEP_DEG = 15.0
SCAN_MAX_DEG = 60.0

JNT_VEL = 20.0
JNT_ACC = 120.0


# =========================
# 1) 책상 안쪽(inside) 서치 패턴
# =========================
def execute_find_scan_step_desk(ctx: MotionContext, step_idx: int):
    """
    책상 위에서 쓸 기존 스윕 패턴:

      - base joint(j1)를 좌/우로 번갈아가며 회전
      - step_idx:
          0 -> +SCAN_STEP_DEG
          1 -> -SCAN_STEP_DEG
          2 -> +2*SCAN_STEP_DEG
          3 -> -2*SCAN_STEP_DEG
          ...
      - 최대 각도는 ±SCAN_MAX_DEG로 클램프
      - J5는 안쪽에서 바깥쪽 방향으로 탐색
    """
    from DSR_ROBOT2 import (
        posj,
        DR_MV_MOD_REL,
        DR_MV_RA_DUPLICATE,
    )

    if ctx.is_cancelled():
        ctx.node.get_logger().warn("[FIND-DESK] cancel detected before scan step")
        raise MotionCancelled("Cancelled before desk scan step")

    k = step_idx // 2 + 1
    sign = 1 if (step_idx % 2 == 0) else -1  # 짝수: +, 홀수: -

    delta_x = sign * k * SCAN_STEP_DEG
    delta_y = -0.5 * SCAN_STEP_DEG

    target_j = posj(delta_x, 0, 0, 0, delta_y)

    ctx.motion.movej(
        target_j,
        vel=JNT_VEL,
        acc=JNT_ACC,
        mod=DR_MV_MOD_REL,
        ra=DR_MV_RA_DUPLICATE,
    )


# =========================
# 2) 책상 바깥(outside) 서치 패턴
# =========================
def execute_find_scan_step_outside(ctx: MotionContext, step_idx: int):
    """
    책상 바깥(사람 / 의자 등)을 찾기 위한 패턴.

    요구사항:
      - 시작 자세: posj(30, -30, 130, 90, 0, 0)
      - J1을 -90도까지 천천히 마이너스 방향으로 스캔,
        끝에 도달하면 +30도까지 플러스 방향으로 스캔 (왕복)
      - 편도 1회(한쪽 끝 → 반대쪽 끝)를 돌 때마다 J2를 -5도씩 더 숙임
      - J2의 최소각도는 -45도

    구현 방식:
      - step_idx를 이용해 수학적으로 (J1, J2)를 결정하는 stateless 패턴
      - 각 step은 ABS movej로 이동
    """
    from DSR_ROBOT2 import (
        posj,
        DR_MV_MOD_ABS,
        DR_MV_RA_DUPLICATE,
    )

    if ctx.is_cancelled():
        ctx.node.get_logger().warn("[FIND-OUTSIDE] cancel detected before scan step")
        raise MotionCancelled("Cancelled before outside scan step")

    # 한 번의 왕복 라인을 얼마나 쪼갤지 (수평 분해능)
    H_STEPS = 8  # 수평 방향 step 개수 (원하면 조절 가능)

    j1_min = -90.0
    j1_max = 30.0
    width = j1_max - j1_min  # 120deg

    sweep_idx = step_idx // H_STEPS      # 몇 번째 왕복 라인인지 (0,1,2,...)
    horiz_idx = step_idx % H_STEPS       # 현재 라인에서 몇 번째 수평 포인트인지 (0..H_STEPS-1)

    # 짝수번째 sweep: J1을 +30 → -90 (마이너스 방향)
    # 홀수번째 sweep: J1을 -90 → +30 (플러스 방향)
    direction = -1 if (sweep_idx % 2 == 0) else 1

    # 0.0 ~ 1.0 사이 보간 인자
    t = 0.0
    if H_STEPS > 1:
        t = horiz_idx / float(H_STEPS - 1)

    if direction == -1:
        # +30 → -90
        j1 = j1_max - width * t
    else:
        # -90 → +30
        j1 = j1_min + width * t

    # J2는 sweep(편도 1회)마다 -5도씩 더 숙임, 최소 -45도
    j2 = -30.0 - 5.0 * sweep_idx
    if j2 < -45.0:
        j2 = -45.0

    # 나머지 관절은 시작자세 기준 유지
    j3 = 130.0
    j4 = 90.0
    j5 = 0.0
    j6 = 0.0

    target_j = posj(j1, j2, j3, j4, j5, j6)

    ctx.node.get_logger().info(
        f"[FIND-OUTSIDE] step={step_idx}, sweep={sweep_idx}, horiz={horiz_idx}, "
        f"j1={j1:.1f}, j2={j2:.1f}"
    )

    ctx.motion.movej(
        target_j,
        vel=JNT_VEL,
        acc=JNT_ACC,
        mod=DR_MV_MOD_ABS,
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
#     params_json: '{\"max_search_time\": 20.0, \"scan_interval\": 0.5, \"search_region\": \"outside\"}'
#   }
# }"


def run_find_skill(
    cmd: SkillCommand,
    ctx: MotionContext,
) -> Tuple[bool, str, float, PoseStamped]:
    """
    FIND 스킬:

      - object_name에 대해 get_object_pose()를 반복 호출
      - 디텍션 안 되면 스윕 모션(책상 안/밖 패턴 중 하나) 수행
      - conf >= FIND_CONF_TH 인 디텍션이 나오면 그 즉시 성공 반환
      - 포즈는 여기서 안 쓰고, 다음 스킬(Pick 등)에서 다시 perception 호출해서 쓰도록 설계

    params_json 예시:
      {
        "max_search_time": 10.0,         # seconds (default 30)
        "scan_interval": 1.0,           # seconds, 모션 스텝 간격
        "search_region": "desk" | "outside"  # 탐색 영역 (기본값: "desk")
      }

    return:
      success, message, confidence, final_pose
      - final_pose: Find는 "자세 찾기" 용이라 여기서는 빈 PoseStamped() 반환
    """
    from DSR_ROBOT2 import (
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
    max_search_time = 30.0
    scan_interval = 1.0
    search_region = "desk"   # "desk" | "outside"

    # params_json 파싱
    if params_json:
        try:
            params = json.loads(params_json)
            max_search_time = float(params.get("max_search_time", max_search_time))
            scan_interval = float(params.get("scan_interval", scan_interval))
            search_region = params.get("search_region", search_region)
            ctx.node.get_logger().info(
                f"[FIND] params: max_search_time={max_search_time}, "
                f"scan_interval={scan_interval}, "
                f"search_region={search_region}"
            )
        except json.JSONDecodeError:
            ctx.node.get_logger().warn(
                f"[FIND] params_json 파싱 실패: {params_json}"
            )

    ctx.node.get_logger().info(
        f"[FIND] skill 실행: object_name='{object_name}', "
        f"max_search_time={max_search_time}s, "
        f"search_region={search_region}"
    )

    if ctx.is_cancelled():
        raise MotionCancelled("Cancelled before FIND start")

    start_time = time.time()
    last_scan_time = start_time - scan_interval  # 시작하자마자 한 번 움직이게
    step_idx = 0

    # 시작자세: desk vs outside
    if search_region == "outside":
        START_POSE = posj(30.0, -30.0, 130.0, 90.0, 0.0, 0.0)
    else:
        START_POSE = posj(0.0, 0.0, 90.0, 0.0, 120.0, 90.0)

    if ctx.is_cancelled():
        raise MotionCancelled("Cancelled before moving to START_POSE")

    ctx.motion.movej(
        START_POSE,
        vel=JNT_VEL,
        acc=JNT_ACC,
        mod=DR_MV_MOD_ABS,
        ra=DR_MV_RA_DUPLICATE,
    )

    while True:
        if ctx.is_cancelled():
            ctx.node.get_logger().warn("[FIND] cancel flag detected, aborting FIND")
            raise MotionCancelled("FIND cancelled by user request")

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

        if ctx.is_cancelled():
            raise MotionCancelled("Cancelled before get_object_pose")

        # 2) 현재 자세에서 object pose 시도
        pose_resp = ctx.request_object_pose(object_name)
        if pose_resp is not None and pose_resp.success:
            conf = float(pose_resp.confidence)
            if conf >= FIND_CONF_TH:
                ctx.node.get_logger().info(
                    f"[FIND] object detected: conf={conf:.2f} >= {FIND_CONF_TH:.2f}, 탐색 종료"
                )
                msg = "object found"
                return True, msg, conf, PoseStamped()
            else:
                ctx.node.get_logger().info(
                    f"[FIND] detected but conf={conf:.2f} < TH={FIND_CONF_TH:.2f}, 계속 탐색"
                )

        if ctx.is_cancelled():
            raise MotionCancelled("Cancelled before get_object_pose")

        # 3) 스윕 모션: 일정 시간마다 한 스텝씩 자세 변경
        if now - last_scan_time >= scan_interval:
            ctx.node.get_logger().info(
                f"[FIND] scan step {step_idx} (elapsed={elapsed:.1f}s)"
            )
            try:
                if search_region == "outside":
                    execute_find_scan_step_outside(ctx, step_idx)
                else:
                    execute_find_scan_step_desk(ctx, step_idx)
            except Exception as e:
                ctx.node.get_logger().error(f"[FIND] scan step error: {e}")
            step_idx += 1
            last_scan_time = now

        # 4) 너무 바쁘게 돌지 않게 조금 쉼
        time.sleep(0.1)
        if ctx.is_cancelled():
            ctx.node.get_logger().warn("[FIND] cancel during sleep")
            raise MotionCancelled("FIND cancelled during sleep")
