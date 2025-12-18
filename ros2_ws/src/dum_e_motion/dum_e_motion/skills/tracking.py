# dum_e_motion/skills/tracking.py

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
# 회전 제어 (시선 맞추기)
ANG_GAIN = 0.7               # 회전 반응 속도 (0~1)
ANG_MAX_STEP_DEG = 10.0      # 한 번에 회전할 최대 각도 [deg]
ANG_DEADZONE_DEG = 1.0       # 회전 오차 허용 범위 [deg]

# Z 거리 제어 (카메라-타겟 거리)
TARGET_DIST_Z = 500.0        # 유지하고 싶은 카메라-타겟 거리 [mm]
DIST_TOL_Z = 50.0            # 이 범위 안이면 Z 이동 X [mm]
MAX_FOLLOW_DIST_Z = 900.0    # 이보다 멀면 Z로는 더 이상 쫓지 않음 [mm]
Z_GAIN = 0.3                 # Z 방향 반응 속도
Z_MAX_STEP = 50.0            # 한 번에 이동할 최대 Z 스텝(mm, 카메라 프레임 기준)

# 기타
LOOP_DT = 0.02               # 루프 주기 [sec] (50Hz)
MAX_TIMEOUT = 5.0            # 물체를 이 시간 이상 못 보면 종료 [sec]
MIN_Z = 200.0                # 바닥 충돌 방지 높이 [mm]


def _normalize(v: np.ndarray, eps: float = 1e-9):
    n = float(np.linalg.norm(v))
    if n < eps:
        return None
    return v / n


def _rot_from_two_vec(a_unit: np.ndarray, b_unit: np.ndarray) -> np.ndarray:
    """
    unit vector a -> unit vector b 로 보내는 최소 회전 (Rodrigues + 특수 케이스 처리)
    반환: 3x3 rotation matrix
    """
    v = np.cross(a_unit, b_unit)
    c = float(np.dot(a_unit, b_unit))
    s = float(np.linalg.norm(v))

    # 거의 동일한 방향 (0도 근처)
    if s < 1e-9 and c > 0:
        return np.eye(3)

    # 정반대 방향 (180도 근처)
    if s < 1e-9 and c < 0:
        # a에 수직인 아무 축이나 180도 회전축으로 사용
        axis = np.array([1.0, 0.0, 0.0], dtype=float)
        if abs(a_unit[0]) > 0.9:
            axis = np.array([0.0, 1.0, 0.0], dtype=float)
        axis = axis - a_unit * float(np.dot(a_unit, axis))
        axis = _normalize(axis)
        if axis is None:
            axis = np.array([0.0, 0.0, 1.0], dtype=float)
        return R.from_rotvec(np.pi * axis).as_matrix()

    # 일반 케이스: Rodrigues
    vx = np.array(
        [
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0],
        ],
        dtype=float,
    )
    Rm = np.eye(3) + vx + (vx @ vx) * ((1.0 - c) / (s * s))
    return Rm


def run_tracking_skill(cmd: SkillCommand, ctx: MotionContext):
    """
    [Tracking v3]
    - 회전: 카메라 시선(+Z)이 타겟을 바라보도록 orientation 제어
    - Z 거리: 카메라-타겟 거리를 500mm 근처로 유지
        - dist < TARGET_DIST_Z - DIST_TOL_Z  → 뒤로 물러남
        - dist > TARGET_DIST_Z + DIST_TOL_Z  → 앞으로 다가감 (단, dist <= MAX_FOLLOW_DIST_Z 까지만)
    - XY는 제어하지 않음 (시선 + Z 거리만)
    """
    from DSR_ROBOT2 import get_current_posx, DR_MV_MOD_ABS

    object_name = cmd.object_name
    ctx.node.get_logger().info(f"[TRACKING] Start tracking v3 (orientation + Z) '{object_name}'")

    # ---------------------------------------------------------
    # 0) gripper2cam 회전 행렬 준비
    #    ctx.gripper2cam: TCP(그리퍼) -> 카메라 변환이라고 가정
    # ---------------------------------------------------------
    ctx.node.get_logger().info(f"[TRACKING][DEBUG] gripper2cam type = {type(ctx.gripper2cam)}")

    if ctx.gripper2cam is None:
        ctx.node.get_logger().warn("[TRACKING] gripper2cam is None. Using identity rotation (TEMP).")
        R_g_c = np.eye(3)
    else:
        try:
            R_g_c = np.array(ctx.gripper2cam[:3, :3], dtype=float)
            if R_g_c.shape != (3, 3):
                raise ValueError(f"R_g_c shape={R_g_c.shape}")
        except Exception as e:
            ctx.node.get_logger().warn(
                f"[TRACKING] Invalid gripper2cam shape/type ({e}). Using identity rotation (TEMP)."
            )
            R_g_c = np.eye(3)

    # cam -> gripper(TCP) 회전
    R_c_g = R_g_c.T

    lost_start_time = None

    while not ctx.is_cancelled():
        # ---------------------------------------------------------
        # 1) Perception: 물체 위치 요청 (Tracking 모드)
        # ---------------------------------------------------------
        resp = ctx.request_object_pose(object_name, use_tracking=True)

        if resp is None:
            ctx.node.get_logger().warn("[TRACKING] get_object_pose returned None (timeout/failure)")
            time.sleep(0.2)
            continue

        if not resp.success:
            if lost_start_time is None:
                lost_start_time = time.time()

            if time.time() - lost_start_time > MAX_TIMEOUT:
                ctx.node.get_logger().warn("[TRACKING] Target lost for too long. Stop tracking.")
                return False, "Target lost for too long", 0.0, PoseStamped()

            ctx.node.get_logger().warn("[TRACKING] Searching target...")
            time.sleep(0.2)
            continue

        lost_start_time = None

        if resp.pose is None:
            ctx.node.get_logger().warn("[TRACKING] resp.pose is None (unexpected).")
            time.sleep(0.1)
            continue

        # ---------------------------------------------------------
        # 2) 카메라 기준 물체 위치 / 방향
        # ---------------------------------------------------------
        pos = resp.pose.pose.position
        v_cam_obj = np.array([pos.x, pos.y, pos.z], dtype=float)

        dir_cam = _normalize(v_cam_obj)
        if dir_cam is None:
            ctx.node.get_logger().warn("[TRACKING] object vector norm too small. Skip.")
            time.sleep(LOOP_DT)
            continue

        dist_cam = float(v_cam_obj[2])  # 카메라 z축 방향 거리 [mm]

        # ---------------------------------------------------------
        # 3) 현재 TCP pose 읽기 (Doosan posx: X,Y,Z,A,B,C / Euler Z-Y'-Z'')
        # ---------------------------------------------------------
        curr_res = get_current_posx()
        if not isinstance(curr_res, (list, tuple)) or len(curr_res) == 0:
            ctx.node.get_logger().warn(f"[TRACKING] get_current_posx invalid: {curr_res}")
            time.sleep(LOOP_DT)
            continue

        curr_pos = curr_res[0]
        if not isinstance(curr_pos, (list, tuple)) or len(curr_pos) < 6:
            ctx.node.get_logger().warn(f"[TRACKING] current posx invalid: {curr_pos}")
            time.sleep(LOOP_DT)
            continue

        xyz = np.array(curr_pos[0:3], dtype=float)
        abc_deg = np.array(curr_pos[3:6], dtype=float)

        try:
            # Doosan 표준: Euler Z-Y'-Z'' (ZYZ)
            R_base_tcp = R.from_euler("ZYZ", abc_deg, degrees=True).as_matrix()
        except Exception as e:
            ctx.node.get_logger().error(f"[TRACKING] Failed to build R_base_tcp from abc={abc_deg}: {e}")
            time.sleep(LOOP_DT)
            continue

        # Z 최소 높이 보장
        if xyz[2] < MIN_Z:
            xyz[2] = MIN_Z

        # =========================================================
        #   A) Z 거리 제어 (카메라 기준 dist_cam 사용)
        # =========================================================
        # 기본: XYZ 그대로, 단 Z만 카메라 z기준으로 앞뒤 이동
        step_z_cam = 0.0

        if dist_cam <= 0:
            # 이상한 값이면 Z 제어 스킵
            pass
        else:
            # 목표 거리보다 너무 가까우면 뒤로 물러남
            if dist_cam < TARGET_DIST_Z - DIST_TOL_Z:
                diff = (dist_cam - TARGET_DIST_Z)  # 음수 (가까움)
                step_z_cam = Z_GAIN * diff        # 음수 → 카메라 기준 -Z 방향 이동
            # 목표 거리보다 멀지만, 따라갈 수 있는 범위 안이면 앞으로 감
            elif dist_cam > TARGET_DIST_Z + DIST_TOL_Z and dist_cam < MAX_FOLLOW_DIST_Z:
                diff = (dist_cam - TARGET_DIST_Z)  # 양수 (멀다)
                step_z_cam = Z_GAIN * diff        # 양수 → 카메라 기준 +Z 방향 이동
            # dist_cam >= MAX_FOLLOW_DIST_Z 이면 Z는 안 따라감 (0)

            # 스텝 제한
            if abs(step_z_cam) > Z_MAX_STEP:
                step_z_cam = np.sign(step_z_cam) * Z_MAX_STEP

        # 카메라 프레임 Z 스텝을 TCP → Base로 변환
        if abs(step_z_cam) > 1e-6:
            step_vec_cam = np.array([0.0, 0.0, step_z_cam], dtype=float)  # cam Δ
            v_tcp = R_c_g @ step_vec_cam                                  # cam → tcp
            v_base = R_base_tcp @ v_tcp                                   # tcp → base
            new_xyz = xyz + v_base
        else:
            new_xyz = xyz.copy()

        # 다시 Z 최소 높이 보장
        if new_xyz[2] < MIN_Z:
            new_xyz[2] = MIN_Z

        # =========================================================
        #   B) 회전 제어 (Orientation Look-at)
        # =========================================================
        # TCP 기준에서의 물체 방향
        dir_tcp = R_c_g @ dir_cam  # cam -> tcp

        # TCP +Z 축이 물체 방향을 바라보도록 회전
        z_tcp = np.array([0.0, 0.0, 1.0], dtype=float)
        R_tcp_delta = _rot_from_two_vec(z_tcp, dir_tcp)

        # Base 기준 목표 TCP 회전
        R_base_tcp_des = R_base_tcp @ R_tcp_delta

        R_curr = R.from_matrix(R_base_tcp)
        R_des = R.from_matrix(R_base_tcp_des)

        # 현재에서 목표까지의 상대 회전
        R_delta = R_curr.inv() * R_des
        rotvec = R_delta.as_rotvec()
        angle_rad = float(np.linalg.norm(rotvec))
        ang_deg = float(np.degrees(angle_rad))

        if ang_deg < ANG_DEADZONE_DEG:
            # 회전 오차가 작으면 그대로 유지
            R_step = R_curr
        else:
            # 한 번에 너무 많이 돌지 않도록 step 제한 + gain 적용
            step_deg = min(ang_deg, ANG_MAX_STEP_DEG) * ANG_GAIN
            step_rad = np.radians(step_deg)

            axis = rotvec / (angle_rad + 1e-9)
            step_rotvec = axis * step_rad

            R_step = R_curr * R.from_rotvec(step_rotvec)

        # 최종 TCP 회전 → Doosan Euler ZYZ (A,B,C)
        try:
            abc_step = R_step.as_euler("ZYZ", degrees=True)
        except Exception as e:
            ctx.node.get_logger().error(f"[TRACKING] Euler conversion failed: {e}")
            time.sleep(0.1)
            continue

        # ---------------------------------------------------------
        # 4) 최종 target pose = [Z distance + orientation] 업데이트
        # ---------------------------------------------------------
        target_pos = [
            float(new_xyz[0]),
            float(new_xyz[1]),
            float(new_xyz[2]),
            float(abc_step[0]),
            float(abc_step[1]),
            float(abc_step[2]),
        ]

        try:
            ctx.motion.movel(
                target_pos,
                vel=ctx.LIN_VEL,
                acc=ctx.LIN_ACC,
                mod=DR_MV_MOD_ABS,
            )
        except MotionCancelled:
            ctx.node.get_logger().warn("[TRACKING] Motion cancelled by user.")
            break
        except Exception as e:
            ctx.node.get_logger().error(f"[TRACKING] Move failed: {e}")
            time.sleep(0.2)

        time.sleep(LOOP_DT)

    # ---------------------------------------------------------
    # 종료 시 홈 복귀
    # ---------------------------------------------------------
    ctx.node.get_logger().info("[TRACKING] Loop finished. Executing Home Motion.")
    execute_home_motion(ctx)
    return True, "Tracking finished", 1.0, PoseStamped()
