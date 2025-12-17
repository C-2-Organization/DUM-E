# dum_e_motion/skills/tracking.py
# Solo-Usage:
# ros2 service call /run_skill dum_e_interfaces/srv/RunSkill "{
#   command: {
#     skill_type: 5,
#     object_name: 'hammer',
#     target_pose: {
#       header: {frame_id: ''},
#       pose: {
#         position: {x: 0.0, y: 0.0, z: 0.0},
#         orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}
#       }
#     },
#     params_json: '{}'
#   }
# }"

import time
import numpy as np
from scipy.spatial.transform import Rotation as R
from geometry_msgs.msg import PoseStamped

from dum_e_interfaces.msg import SkillCommand
from dum_e_motion.motion_context import MotionContext, MotionCancelled
from dum_e_motion.skills.home import execute_home_motion

# ---------------------------------------------------------
# 설정값
# ---------------------------------------------------------
TARGET_DISTANCE = 100.0   # 물체와 유지할 거리 100mm
GAIN = 0.5               # 반응 속도 (너무 빠르면 진동 발생)
MAX_STEP = 100.0         # 한 번에 이동할 최대 거리 100mm (안전장치)
DEADZONE = 20.0          # 오차 허용 범위 20mm (이 안에서는 안 움직임)
MAX_TIMEOUT = 5.0        # 물체가 MAX_TIMEOUT초 이상 안 보일 시 추적 중지


def run_tracking_skill(cmd: SkillCommand, ctx: MotionContext):
    """
    물체를 카메라 중심에 두고 일정 거리를 유지하도록 로봇을 제어함
    """
    from DSR_ROBOT2 import get_current_posx, DR_MV_MOD_ABS

    object_name = cmd.object_name
    ctx.node.get_logger().info(f"[TRACKING] Start tracking '{object_name}'")

    # ---------------------------------------------------------
    # 0) gripper2cam 안전 처리
    # ---------------------------------------------------------
    ctx.node.get_logger().info(
        f"[TRACKING][DEBUG] gripper2cam type = {type(ctx.gripper2cam)}"
    )

    # 1. 그리퍼 -> 카메라 회전 행렬 추출 (None 방어)
    if ctx.gripper2cam is None:
        ctx.node.get_logger().warn(
            "[TRACKING] gripper2cam is None. Using identity rotation (TEMP)."
        )
        R_g_c = np.eye(3)
    else:
        try:
            R_g_c = ctx.gripper2cam[:3, :3]
        except Exception as e:
            ctx.node.get_logger().warn(
                f"[TRACKING] Invalid gripper2cam shape/type ({e}). Using identity rotation (TEMP)."
            )
            R_g_c = np.eye(3)

    lost_start_time = None

    while not ctx.is_cancelled():
        # ---------------------------------------------------------
        # 1) Perception: 물체 위치 요청 (Tracking 모드)
        # ---------------------------------------------------------
        resp = ctx.request_object_pose(object_name, use_tracking=True)

        # 1-1) 서비스 타임아웃/호출 실패
        if resp is None:
            ctx.node.get_logger().warn("[TRACKING] get_object_pose returned None (timeout/failure)")
            time.sleep(0.2)
            continue

        # 1-2) 물체를 못 찾음(success=False)
        if not resp.success:
            if lost_start_time is None:
                lost_start_time = time.time()

            if time.time() - lost_start_time > MAX_TIMEOUT:
                ctx.node.get_logger().warn("[TRACKING] Target lost for too long. Stop tracking.")
                return False, "Target lost for too long", 0.0, PoseStamped()

            ctx.node.get_logger().warn("[TRACKING] Searching...")
            time.sleep(0.2)
            continue

        # success=True로 들어오면 “잃어버림 타이머” 초기화
        lost_start_time = None

        # ---------------------------------------------------------
        # 2) 카메라 기준 물체 위치
        # ---------------------------------------------------------
        # resp.pose가 비정상인 경우 방어
        if resp.pose is None:
            ctx.node.get_logger().warn("[TRACKING] resp.pose is None (unexpected).")
            time.sleep(0.2)
            continue

        pos = resp.pose.pose.position
        v_cam_obj = np.array([pos.x, pos.y, pos.z], dtype=float)

        # ---------------------------------------------------------
        # 3) 오차 계산 (Camera Frame)
        # 목표: 물체가 (0, 0, TARGET_DISTANCE)에 오도록 함
        # ---------------------------------------------------------
        target_in_cam = np.array([0.0, 0.0, TARGET_DISTANCE], dtype=float)
        err_vec_cam = v_cam_obj - target_in_cam  # [dx, dy, dz]

        # 데드존 체크 (오차가 작으면 이동 생략)
        if np.linalg.norm(err_vec_cam) < DEADZONE:
            time.sleep(0.05)
            continue

        # 이동량 제한 (안전장치)
        step_vec_cam = err_vec_cam * GAIN
        step_len = np.linalg.norm(step_vec_cam)
        if step_len > MAX_STEP:
            step_vec_cam = step_vec_cam * (MAX_STEP / step_len)

        # ---------------------------------------------------------
        # 4) 좌표 변환 (Camera Delta -> Base Delta)
        # ---------------------------------------------------------
        curr_res = get_current_posx()
        if curr_res is None or len(curr_res) == 0 or curr_res[0] is None:
            ctx.node.get_logger().warn("[TRACKING] get_current_posx returned invalid result.")
            time.sleep(0.05)
            continue

        curr_pos = curr_res[0]  # [x, y, z, A, B, C] (mm, deg)
        if len(curr_pos) < 6:
            ctx.node.get_logger().warn("[TRACKING] current posx has invalid length.")
            time.sleep(0.05)
            continue

        curr_abc_deg = np.array(curr_pos[3:6], dtype=float)

        # Base -> TCP 회전 행렬 (Euler ZYZ)
        r_base_tcp = R.from_euler('zyz', curr_abc_deg, degrees=True).as_matrix()

        # Camera Frame의 이동 벡터(mm)를 Base Frame(mm)으로 변환
        v_base = r_base_tcp @ R_g_c @ step_vec_cam

        # ---------------------------------------------------------
        # 5) 이동 명령 (Absolute Move, mm)
        # ---------------------------------------------------------
        target_pos = list(curr_pos)
        target_pos[0] += float(v_base[0])
        target_pos[1] += float(v_base[1])
        target_pos[2] += float(v_base[2])

        # 바닥 충돌 방지 (Z 최소 높이 200mm)
        if target_pos[2] < 200.0:
            target_pos[2] = 200.0

        try:
            ctx.motion.movel(
                target_pos,
                vel=ctx.LIN_VEL,
                acc=ctx.LIN_ACC,
                mod=DR_MV_MOD_ABS
            )
        except MotionCancelled:
            # 이동 중 Stop 신호가 들어오면 루프 탈출
            ctx.node.get_logger().warn("[TRACKING] Motion cancelled by user.")
            break
        except Exception as e:
            ctx.node.get_logger().error(f"[TRACKING] Move failed: {e}")
            time.sleep(0.2)

    # ---------------------------------------------------------
    # [수정] 반복문 종료 후(또는 취소 시) 무조건 홈 복귀
    # ---------------------------------------------------------
    ctx.node.get_logger().info("[TRACKING] Loop finished. Executing Home Motion.")
    
    execute_home_motion(ctx)

    return True, "Tracking finished", 1.0, PoseStamped()
