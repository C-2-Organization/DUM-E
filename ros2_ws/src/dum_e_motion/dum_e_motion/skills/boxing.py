# dum_e_motion/skills/boxing.py

import time
import random
import numpy as np
from scipy.spatial.transform import Rotation as R
from geometry_msgs.msg import PoseStamped

from dum_e_interfaces.msg import SkillCommand
from dum_e_motion.motion_context import MotionContext, MotionCancelled
from dum_e_motion.skills.home import execute_home_motion

# ---------------------------------------------------------
# 기본 트래킹 설정값 (tracking.py와 거의 동일)
# ---------------------------------------------------------
ANG_GAIN = 0.7               # 회전 반응 속도 (0~1)
ANG_MAX_STEP_DEG = 10.0      # 한 번에 회전할 최대 각도 [deg]
ANG_DEADZONE_DEG = 1.0       # 회전 오차 허용 범위 [deg]

TARGET_DIST_Z = 500.0        # 유지하고 싶은 카메라-타겟 거리 [mm]
DIST_TOL_Z = 50.0            # 이 범위 안이면 Z 이동 X [mm]
MAX_FOLLOW_DIST_Z = 900.0    # 이보다 멀면 Z로는 더 이상 쫓지 않음 [mm]
Z_GAIN = 0.3                 # Z 방향 반응 속도
Z_MAX_STEP = 50.0            # 한 번에 이동할 최대 Z 스텝(mm, 카메라 프레임 기준)

LOOP_DT = 0.02               # 메인 루프 주기 [sec]
MAX_TIMEOUT = 5.0            # 물체를 이 시간 이상 못 보면 종료 [sec]
MIN_Z = 200.0                # 바닥 충돌 방지 높이 [mm]

# ---------------------------------------------------------
# BOXING 전용 설정값
# ---------------------------------------------------------
# ⚠ 기존 5000mm는 거의 무조건 워크스페이스 밖으로 나감
JAB_FORWARD_MM = 120.0       # 기본 잽 목표 길이 [mm]
SAFE_FACE_MARGIN = 200.0     # 얼굴 앞에서 최소로 남겨둘 거리 [mm]
MAX_JAB_STEP = 150.0         # 한 번에 허용하는 최대 잽 길이 [mm]

JAB_BACK_RATIO = 1.0         # 잽 후 원래 위치로 얼마나 되돌아올지 (1.0 = 완전 복귀)
JAB_HOLD_SEC = 0.1           # 잽 자세로 잠깐 유지하는 시간 [sec]

JAB_MIN_INTERVAL = 1.0       # 최소 잽 간격 [sec]
JAB_MAX_INTERVAL = 2.5       # 최대 잽 간격 [sec]

# 움직임 속도/가속도 (가드/잽/복귀 모두 동일하게 사용)
VEL = [250, 500]
ACC = [250, 400]

MAX_BOXING_DURATION = 180.0  # 전체 BOXING 스킬 최대 지속시간 [sec]


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


def _sample_next_jab_interval() -> float:
    """다음 잽까지 기다릴 시간을 [JAB_MIN_INTERVAL, JAB_MAX_INTERVAL]에서 랜덤 샘플."""
    return random.uniform(JAB_MIN_INTERVAL, JAB_MAX_INTERVAL)


def run_boxing_skill(cmd: SkillCommand, ctx: MotionContext):
    """
    BOXING 스킬

    - 기본적으로 얼굴(또는 사람)을 트래킹하면서
      일정 간격으로 얼굴 방향으로 팔(그리퍼)을 뻗는 잽 동작을 실행.
    - Tracking v3 코드 기반:
      * 카메라 시선(+Z)이 타겟을 바라보도록 orientation 제어
      * 카메라-타겟 Z 거리 제어 (TARGET_DIST_Z 근처 유지)
    - 추가로:
      * 주기적으로 타겟 방향으로 일정 거리만큼 손을 내밀었다가
        다시 원래 guard 자세로 복귀.
    """

    from DSR_ROBOT2 import get_current_posx, DR_MV_MOD_ABS, posx

    object_name = cmd.object_name.strip() if cmd.object_name else "face"
    ctx.node.get_logger().info(
        f"[BOXING] Start BOXING skill (tracking + jab) target='{object_name}'"
    )

    # ---------------------------------------------------------
    # 0) gripper2cam 회전 행렬 준비
    # ---------------------------------------------------------
    ctx.node.get_logger().info(f"[BOXING][DEBUG] gripper2cam type = {type(ctx.gripper2cam)}")

    if ctx.gripper2cam is None:
        ctx.node.get_logger().warn("[BOXING] gripper2cam is None. Using identity rotation (TEMP).")
        R_g_c = np.eye(3)
    else:
        try:
            R_g_c = np.array(ctx.gripper2cam[:3, :3], dtype=float)
            if R_g_c.shape != (3, 3):
                raise ValueError(f"R_g_c shape={R_g_c.shape}")
        except Exception as e:
            ctx.node.get_logger().warn(
                f"[BOXING] Invalid gripper2cam shape/type ({e}). Using identity rotation (TEMP)."
            )
            R_g_c = np.eye(3)

    # cam -> gripper(TCP) 회전
    R_c_g = R_g_c.T

    lost_start_time = None

    # BOXING 전체 시간 관리
    skill_start_time = time.time()

    # 잽 타이밍 관리
    last_jab_time = time.time()
    next_jab_interval = _sample_next_jab_interval()

    # 직전 "가드 자세" (잽을 뻗기 전의 TCP pose)
    guard_pos = None  # [x,y,z,a,b,c]

    # 시작 위치: 복싱 링으로 이동
    ctx.motion.close_gripper()
    init_pos = posx(450, -350, 480, 90, -90, -90)
    ctx.node.get_logger().info("복싱링으로 이동합니다.")
    try:
        ctx.motion.movel(
            init_pos,
            vel=VEL,
            acc=ACC,
            mod=DR_MV_MOD_ABS,
        )
    except MotionCancelled:
        ctx.node.get_logger().warn("[BOXING] Motion cancelled while moving to boxing ring.")
        execute_home_motion(ctx)
        return False, "Cancelled before boxing start", 0.0, PoseStamped()
    except Exception as e:
        ctx.node.get_logger().error(f"[BOXING] Failed to move to initial boxing pose: {e}")
        execute_home_motion(ctx)
        return False, f"Init move failed: {e}", 0.0, PoseStamped()

    while not ctx.is_cancelled():
        now = time.time()
        elapsed = now - skill_start_time
        if elapsed > MAX_BOXING_DURATION:
            ctx.node.get_logger().info("[BOXING] Max boxing duration reached. Stopping.")
            break

        # ---------------------------------------------------------
        # 1) Perception: 타겟 위치 요청 (Tracking 모드)
        # ---------------------------------------------------------
        resp = ctx.request_object_pose(object_name, use_tracking=True)

        if resp is None:
            ctx.node.get_logger().warn("[BOXING] get_object_pose returned None (timeout/failure)")
            time.sleep(0.2)
            continue

        if not resp.success:
            if lost_start_time is None:
                lost_start_time = time.time()

            if time.time() - lost_start_time > MAX_TIMEOUT:
                ctx.node.get_logger().warn("[BOXING] Target lost for too long. Stop boxing.")
                return False, "Target lost for too long", 0.0, PoseStamped()

            ctx.node.get_logger().warn("[BOXING] Searching target...")
            time.sleep(0.2)
            continue

        lost_start_time = None

        if resp.pose is None:
            ctx.node.get_logger().warn("[BOXING] resp.pose is None (unexpected).")
            time.sleep(0.1)
            continue

        # ---------------------------------------------------------
        # 2) 카메라 기준 타겟(얼굴) 위치 / 방향
        # ---------------------------------------------------------
        pos = resp.pose.pose.position
        v_cam_obj = np.array([pos.x, pos.y, pos.z], dtype=float)

        dir_cam = _normalize(v_cam_obj)
        if dir_cam is None:
            ctx.node.get_logger().warn("[BOXING] object vector norm too small. Skip.")
            time.sleep(LOOP_DT)
            continue

        dist_cam = float(v_cam_obj[2])  # 카메라 z축 방향 거리 [mm]

        # ---------------------------------------------------------
        # 3) 현재 TCP pose 읽기 (Doosan posx: X,Y,Z,A,B,C / Euler Z-Y'-Z'')
        # ---------------------------------------------------------
        curr_res = get_current_posx()
        if not isinstance(curr_res, (list, tuple)) or len(curr_res) == 0:
            ctx.node.get_logger().warn(f"[BOXING] get_current_posx invalid: {curr_res}")
            time.sleep(LOOP_DT)
            continue

        curr_pos = curr_res[0]
        if not isinstance(curr_pos, (list, tuple)) or len(curr_pos) < 6:
            ctx.node.get_logger().warn(f"[BOXING] current posx invalid: {curr_pos}")
            time.sleep(LOOP_DT)
            continue

        xyz = np.array(curr_pos[0:3], dtype=float)
        abc_deg = np.array(curr_pos[3:6], dtype=float)

        try:
            # Doosan 표준: Euler Z-Y'-Z'' (ZYZ)
            R_base_tcp = R.from_euler("ZYZ", abc_deg, degrees=True).as_matrix()
        except Exception as e:
            ctx.node.get_logger().error(f"[BOXING] Failed to build R_base_tcp from abc={abc_deg}: {e}")
            time.sleep(LOOP_DT)
            continue

        # Z 최소 높이 보장
        if xyz[2] < MIN_Z:
            xyz[2] = MIN_Z

        # =========================================================
        #   A) Z 거리 제어 (Tracking 과 동일)
        # =========================================================
        step_z_cam = 0.0

        if dist_cam > 0:
            if dist_cam < TARGET_DIST_Z - DIST_TOL_Z:
                diff = (dist_cam - TARGET_DIST_Z)  # 음수 (가까움)
                step_z_cam = Z_GAIN * diff
            elif dist_cam > TARGET_DIST_Z + DIST_TOL_Z and dist_cam < MAX_FOLLOW_DIST_Z:
                diff = (dist_cam - TARGET_DIST_Z)  # 양수 (멀다)
                step_z_cam = Z_GAIN * diff

            # 스텝 제한
            if abs(step_z_cam) > Z_MAX_STEP:
                step_z_cam = np.sign(step_z_cam) * Z_MAX_STEP

        if abs(step_z_cam) > 1e-6:
            step_vec_cam = np.array([0.0, 0.0, step_z_cam], dtype=float)
            v_tcp_step = R_c_g @ step_vec_cam
            v_base_step = R_base_tcp @ v_tcp_step
            new_xyz = xyz + v_base_step
        else:
            new_xyz = xyz.copy()

        if new_xyz[2] < MIN_Z:
            new_xyz[2] = MIN_Z

        # =========================================================
        #   B) 회전 제어 (얼굴을 바라보도록 orientation 맞추기)
        # =========================================================
        dir_tcp = R_c_g @ dir_cam  # cam -> tcp

        z_tcp = np.array([0.0, 0.0, 1.0], dtype=float)
        R_tcp_delta = _rot_from_two_vec(z_tcp, dir_tcp)

        R_base_tcp_des = R_base_tcp @ R_tcp_delta

        R_curr = R.from_matrix(R_base_tcp)
        R_des = R.from_matrix(R_base_tcp_des)

        R_delta = R_curr.inv() * R_des
        rotvec = R_delta.as_rotvec()
        angle_rad = float(np.linalg.norm(rotvec))
        ang_deg = float(np.degrees(angle_rad))

        if ang_deg < ANG_DEADZONE_DEG:
            R_step = R_curr
        else:
            step_deg = min(ang_deg, ANG_MAX_STEP_DEG) * ANG_GAIN
            step_rad = np.radians(step_deg)

            axis = rotvec / (angle_rad + 1e-9)
            step_rotvec = axis * step_rad

            R_step = R_curr * R.from_rotvec(step_rotvec)

        try:
            abc_step = R_step.as_euler("ZYZ", degrees=True)
        except Exception as e:
            ctx.node.get_logger().error(f"[BOXING] Euler conversion failed: {e}")
            time.sleep(0.1)
            continue

        # ---------------------------------------------------------
        #   C) 기본 가드 자세(target_pos_guard) 계산
        # ---------------------------------------------------------
        target_pos_guard = [
            float(new_xyz[0]),
            float(new_xyz[1]),
            float(new_xyz[2]),
            float(abc_step[0]),
            float(abc_step[1]),
            float(abc_step[2]),
        ]

        # 처음 루프에서 guard_pos 초기화
        if guard_pos is None:
            guard_pos = target_pos_guard.copy()

        # ---------------------------------------------------------
        #   D) 일단 가드 자세로 이동 (미세 트래킹)
        # ---------------------------------------------------------
        try:
            ctx.motion.movel(
                target_pos_guard,
                vel=VEL,
                acc=ACC,
                mod=DR_MV_MOD_ABS,
            )
        except MotionCancelled:
            ctx.node.get_logger().warn("[BOXING] Motion cancelled while moving to guard.")
            break
        except Exception as e:
            ctx.node.get_logger().error(f"[BOXING] Move to guard failed: {e}")
            time.sleep(0.1)

        guard_pos = target_pos_guard.copy()

        # ---------------------------------------------------------
        #   E) 잽 타이밍 체크 & 실행
        # ---------------------------------------------------------
        now = time.time()
        if now - last_jab_time >= next_jab_interval:
            # 현재 얼굴 거리 dist_cam 기준으로 안전한 잽 길이 계산
            # 얼굴에서 SAFE_FACE_MARGIN 만큼은 남겨두기
            max_allow_jab = max(0.0, dist_cam - SAFE_FACE_MARGIN)
            jab_len = min(JAB_FORWARD_MM, MAX_JAB_STEP, max_allow_jab)

            if jab_len < 1e-3:
                ctx.node.get_logger().warn(
                    f"[BOXING] Too close to target (dist={dist_cam:.1f}mm), skip jab."
                )
            else:
                # 1) 카메라 기준으로 타겟 방향(dir_cam)을 따라 jab_len 만큼 전진
                jab_step_cam = dir_cam * jab_len          # cam 프레임 Δ
                v_tcp_jab = R_c_g @ jab_step_cam          # cam → tcp
                v_base_jab = R_base_tcp @ v_tcp_jab       # tcp → base

                jab_xyz = new_xyz + v_base_jab

                if jab_xyz[2] < MIN_Z:
                    jab_xyz[2] = MIN_Z

                target_pos_jab = [
                    float(jab_xyz[0]),
                    float(jab_xyz[1]),
                    float(jab_xyz[2]),
                    float(abc_step[0]),
                    float(abc_step[1]),
                    float(abc_step[2]),
                ]

                ctx.node.get_logger().info(
                    f"[BOXING] 🥊JAB! interval={next_jab_interval:.2f}s, "
                    f"dist_cam={dist_cam:.1f}mm, jab_len={jab_len:.1f}mm, "
                    f"guard={guard_pos[:3]}, jab={target_pos_jab[:3]}"
                )

                # 2) 잽 자세로 전진
                try:
                    ctx.motion.movel(
                        target_pos_jab,
                        vel=VEL,
                        acc=ACC,
                        mod=DR_MV_MOD_ABS,
                    )
                except MotionCancelled:
                    ctx.node.get_logger().warn("[BOXING] Motion cancelled during jab.")
                    break
                except Exception as e:
                    ctx.node.get_logger().error(f"[BOXING] Move (jab) failed: {e}")
                    # 잽 실패 시에도 일단 루프 계속
                    time.sleep(0.1)

                # 3) 잠깐 유지
                time.sleep(JAB_HOLD_SEC)

                # 4) 가드 자세로 복귀 (JAB_BACK_RATIO 비율만큼)
                if JAB_BACK_RATIO > 0.0:
                    back_xyz = [
                        guard_pos[0] * JAB_BACK_RATIO + target_pos_jab[0] * (1.0 - JAB_BACK_RATIO),
                        guard_pos[1] * JAB_BACK_RATIO + target_pos_jab[1] * (1.0 - JAB_BACK_RATIO),
                        guard_pos[2] * JAB_BACK_RATIO + target_pos_jab[2] * (1.0 - JAB_BACK_RATIO),
                    ]

                    target_pos_back = [
                        float(back_xyz[0]),
                        float(back_xyz[1]),
                        float(back_xyz[2]),
                        float(guard_pos[3]),
                        float(guard_pos[4]),
                        float(guard_pos[5]),
                    ]

                    try:
                        ctx.motion.movel(
                            target_pos_back,
                            vel=VEL,
                            acc=ACC,
                            mod=DR_MV_MOD_ABS,
                        )
                    except MotionCancelled:
                        ctx.node.get_logger().warn("[BOXING] Motion cancelled during back-to-guard.")
                        break
                    except Exception as e:
                        ctx.node.get_logger().error(f"[BOXING] Move (back to guard) failed: {e}")
                        time.sleep(0.1)

                    # 실제 guard_pos를 back 위치로 업데이트
                    guard_pos = target_pos_back.copy()

            # 잽 타이머 리셋
            last_jab_time = time.time()
            next_jab_interval = _sample_next_jab_interval()

        # 메인 루프 슬립
        time.sleep(LOOP_DT)

    # ---------------------------------------------------------
    # 종료 시 홈 복귀
    # ---------------------------------------------------------
    ctx.node.get_logger().info("[BOXING] Loop finished. Executing Home Motion.")
    execute_home_motion(ctx)
    return True, "Boxing finished", 1.0, PoseStamped()
