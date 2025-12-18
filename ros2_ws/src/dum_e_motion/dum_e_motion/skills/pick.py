# dum_e_motion/skills/pick.py
import math
from typing import Tuple

from geometry_msgs.msg import PoseStamped

from dum_e_interfaces.msg import SkillCommand
from dum_e_motion.motion_context import MotionContext


# ==============================================================================
# 설정 및 상수
# ==============================================================================
PICK_CONF_TH = 0.3
GRIPPER_OFFSET = 215  # mm, 그리퍼 오프셋

# ==============================================================================
# Helper function
# ==============================================================================

def _has_valid_external_pose(ps: PoseStamped) -> bool:
    """
    외부에서 실제 의미 있는 target_pose를 넣어준 경우에만 True.
    - frame_id가 비어있지 않고
    - position이 완전히 (0,0,0)인 기본값이 아닐 때만 인정.
    """
    if ps is None:
        return False

    if not ps.header.frame_id:
        return False

    p = ps.pose.position
    if abs(p.x) < 1e-6 and abs(p.y) < 1e-6 and abs(p.z) < 1e-6:
        # 기본 생성된 PoseStamped()라고 판단
        return False

    return True


# ==============================================================================
# 휴리스틱 기반 Grasp 계산
# ==============================================================================
def compute_grasp_from_bbox(
    bbox_norm: Tuple[float, float, float, float],
    pose: PoseStamped,
    logger
) -> Tuple[PoseStamped, float]:
    """
    bbox와 중심 pose로부터 grasp pose와 각도 계산

    휴리스틱:
    1. 위치: bbox 중심의 depth 좌표 사용
    2. 각도: bbox의 짧은 축에 수직 (긴 축 방향으로 그리퍼 정렬)

    Returns:
        (grasp_pose, angle_rad)
    """
    x1, y1, x2, y2 = bbox_norm

    # bbox 크기 계산 (normalized 좌표)
    bbox_w = x2 - x1
    bbox_h = y2 - y1

    # 짧은 축 방향에 수직 = 긴 축 방향으로 그리퍼 정렬
    # bbox가 가로로 긴 경우 (w > h): 그리퍼를 가로 방향 (0도)
    # bbox가 세로로 긴 경우 (h > w): 그리퍼를 세로 방향 (90도)
    if bbox_w >= bbox_h:
        angle_rad = 0.0  # 가로 방향
    else:
        angle_rad = math.pi / 2  # 세로 방향 (90도)

    logger.info(
        f"[HEURISTIC] bbox size: w={bbox_w:.3f}, h={bbox_h:.3f}, "
        f"angle={math.degrees(angle_rad):.1f}°"
    )

    # pose는 그대로 사용 (Perception에서 받은 중심점)
    grasp_pose = PoseStamped()
    grasp_pose.header = pose.header
    grasp_pose.pose.position = pose.pose.position

    # orientation: Z축 회전만 적용
    grasp_pose.pose.orientation.x = 0.0
    grasp_pose.pose.orientation.y = 0.0
    grasp_pose.pose.orientation.z = math.sin(angle_rad / 2.0)
    grasp_pose.pose.orientation.w = math.cos(angle_rad / 2.0)

    return grasp_pose, angle_rad


def normalize_angle(angle_deg: float) -> float:
    """각도를 -180 ~ 180 범위로 정규화"""
    while angle_deg > 180:
        angle_deg -= 360
    while angle_deg < -180:
        angle_deg += 360
    return angle_deg


# ==============================================================================
# 로봇 모션 실행 함수
# ==============================================================================
def execute_pick_motion(ctx: MotionContext, x, y, z, target_yaw_rad=None):
    from DSR_ROBOT2 import DR_MV_MOD_ABS, DR_MV_RA_DUPLICATE, get_current_posx
    from DR_common2 import posx

    ctx.node.get_logger().info(f"[MOVE] Pick → base({x:.3f}, {y:.3f}, {z:.3f})")

    current_pos = get_current_posx()[0]

    # 목표 회전값 (Rz)
    if target_yaw_rad is not None:
        raw_rz = math.degrees(target_yaw_rad)
        next_rz = normalize_angle(raw_rz)
        ctx.node.get_logger().info(f"[MOVE] Gripper angle: {next_rz:.1f}°")
    else:
        next_rz = current_pos[5]

    # 1. 그리퍼 오픈
    ctx.motion.open_gripper()
    ctx.motion.wait(0.5)

    approach_pos = posx([x, y, z, current_pos[3], current_pos[4], next_rz])

    # 2. 접근
    ctx.motion.movel(approach_pos, vel=ctx.LIN_VEL, acc=ctx.LIN_ACC, mod=DR_MV_MOD_ABS, ra=DR_MV_RA_DUPLICATE)

    # 3. 집기
    ctx.motion.close_gripper()
    ctx.motion.wait(0.8)

    # 4. 들어 올리기
    lift_pos = list(approach_pos)
    lift_pos[2] += 100  # 100mm 상승
    lift_pos = posx(lift_pos)
    ctx.motion.movel(lift_pos, vel=ctx.LIN_VEL, acc=ctx.LIN_ACC, mod=DR_MV_MOD_ABS, ra=DR_MV_RA_DUPLICATE)

    # 5. 홈 이동
    ctx.motion.movej(ctx.CUSTOM_HOME_JOINT, vel=ctx.JNT_VEL, acc=ctx.JNT_ACC, mod=DR_MV_MOD_ABS, ra=DR_MV_RA_DUPLICATE)


# ==============================================================================
# 메인 스킬 함수 (Entry Point)
# ==============================================================================
def run_pick_skill(cmd: SkillCommand, ctx: MotionContext) -> Tuple[bool, str, float, PoseStamped]:
    object_name = cmd.object_name.strip()

    if not object_name:
        return False, "object_name is empty", 0.0, PoseStamped()

    ctx.node.get_logger().info(f"[PICK] Start Skill: '{object_name}'")

    cam_pose = None
    confidence = 0.0
    grasp_angle_rad = 0.0

    # ------------------------------------------------------------------
    # CASE A: 외부 Target Pose 사용 (직접 좌표 지정)
    # ------------------------------------------------------------------
    if _has_valid_external_pose(cmd.target_pose):
        cam_pose = cmd.target_pose
        confidence = 1.0
        grasp_angle_rad = 0.0
        ctx.node.get_logger().info(
            f"[PICK] Using external target pose (frame_id={cmd.target_pose.header.frame_id})"
        )

    # ------------------------------------------------------------------
    # CASE B: Perception + Heuristic 기반 (기본 모드)
    # ------------------------------------------------------------------
    else:
        pose_resp = ctx.request_object_pose(object_name, use_tracking=False)
        if not pose_resp or not pose_resp.success:
            return False, f"Perception failed for '{object_name}'", 0.0, PoseStamped()

        confidence = float(pose_resp.confidence)
        if confidence < PICK_CONF_TH:
            return False, f"Low confidence {confidence:.2f}", confidence, PoseStamped()

        # bbox가 있으면 휴리스틱으로 각도 계산
        has_bbox = (
            hasattr(pose_resp, 'bbox_norm')
            and pose_resp.bbox_norm is not None
            and len(pose_resp.bbox_norm) == 4
        )

        if has_bbox:
            bbox_norm = tuple(pose_resp.bbox_norm)
            cam_pose, grasp_angle_rad = compute_grasp_from_bbox(
                bbox_norm,
                pose_resp.pose,
                ctx.node.get_logger()
            )
            ctx.node.get_logger().info(f"[PICK] Heuristic mode for '{object_name}'")
        else:
            # bbox 없으면 각도 0으로
            cam_pose = pose_resp.pose
            grasp_angle_rad = 0.0
            ctx.node.get_logger().warn(f"[PICK] No bbox for '{object_name}', using default angle")

    # ------------------------------------------------------------------
    # 공통: 좌표 변환 및 실행
    # ------------------------------------------------------------------

    # 1. TCP 위치 계산 (그리퍼 오프셋 적용)
    tcp_cam_pose = PoseStamped()
    tcp_cam_pose.header = cam_pose.header
    tcp_cam_pose.pose.position.x = cam_pose.pose.position.x
    tcp_cam_pose.pose.position.y = cam_pose.pose.position.y
    tcp_cam_pose.pose.position.z = cam_pose.pose.position.z - GRIPPER_OFFSET
    tcp_cam_pose.pose.orientation = cam_pose.pose.orientation

    if tcp_cam_pose.pose.position.z <= 50:
        ctx.node.get_logger().warn(
            f"[PICK] Calculated tcp_cam_z={tcp_cam_pose.pose.position.z:.3f} <= 50, abort"
        )
        return False, "Calculated Z is too close/negative", confidence, PoseStamped()

    # 2. Camera -> Base 좌표 변환
    base_xyz = ctx.transform_camera_to_base(tcp_cam_pose)
    bx, by, bz = base_xyz

    # 3. 모션 실행
    try:
        execute_pick_motion(ctx, bx, by, bz, target_yaw_rad=grasp_angle_rad)
        return True, "Success", confidence, ctx.make_final_pose(bx, by, bz)
    except Exception as e:
        ctx.node.get_logger().error(f"[PICK] Motion Error: {e}")
        return False, f"Motion Error: {e}", confidence, PoseStamped()
