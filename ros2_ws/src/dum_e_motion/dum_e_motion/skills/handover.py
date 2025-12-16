# dum_e_motion/skills/handover.py
import json
from typing import Tuple

from geometry_msgs.msg import PoseStamped

from dum_e_interfaces.msg import SkillCommand
from dum_e_motion.motion_context import MotionContext


# 기본 파라미터 (단위: mm / sec)
HANDOVER_CONF_TH = 0.20  # handover는 초기엔 좀 낮게 시작(원하면 올려)
DEFAULT_PRE_X_OFFSET = -80.0   # 손 위치에서 x로 뒤로 80mm (base 기준)
DEFAULT_PRE_Z_OFFSET = 100.0   # 손 위치에서 위로 100mm
DEFAULT_APPROACH_Z_OFFSET = 30.0  # 손 위 30mm까지 접근
DEFAULT_WAIT_SEC = 1.5


def execute_handover_motion(
    ctx: MotionContext,
    x: float,
    y: float,
    z: float,
    *,
    pre_x_offset: float,
    pre_z_offset: float,
    approach_z_offset: float,
    wait_sec: float,
) -> None:
    """
    Handover 모션:
    - prepose(위/뒤) → approach(천천히) → wait → open → retreat(prepose)
    좌표 단위는 Doosan posx와 맞춰 mm 기준.
    """
    from DSR_ROBOT2 import (
        DR_MV_MOD_ABS,
        DR_MV_RA_DUPLICATE,
        get_current_posx,
    )
    from DR_common2 import posx

    ctx.node.get_logger().info(
        f"[HANDOVER] MOVE target base=({x:.3f}, {y:.3f}, {z:.3f})"
    )

    current_pos = get_current_posx()[0]
    rx, ry, rz = current_pos[3], current_pos[4], current_pos[5]

    # prepose (base 기준 오프셋)
    prepose = posx([
        x + pre_x_offset,
        y,
        z + pre_z_offset,
        rx, ry, rz,
    ])

    # approach (손 위쪽 약간 남기고)
    approach = posx([
        x,
        y,
        z + approach_z_offset,
        rx, ry, rz,
    ])

    # 1) prepose
    ctx.motion.movel(
        prepose,
        vel=ctx.LIN_VEL,
        acc=ctx.LIN_ACC,
        mod=DR_MV_MOD_ABS,
        ra=DR_MV_RA_DUPLICATE,
    )

    # 2) approach (사람 앞이니까 느리게)
    ctx.motion.movel(
        approach,
        vel=[50.0, 80.0],
        acc=[50.0, 50.0],
        mod=DR_MV_MOD_ABS,
        ra=DR_MV_RA_DUPLICATE,
    )

    # 3) wait
    ctx.motion.wait(wait_sec)

    # 4) release
    ctx.motion.open_gripper()
    ctx.motion.wait(0.5)

    # 5) retreat
    ctx.motion.movel(
        prepose,
        vel=ctx.LIN_VEL,
        acc=ctx.LIN_ACC,
        mod=DR_MV_MOD_ABS,
        ra=DR_MV_RA_DUPLICATE,
    )


def run_handover_skill(
    cmd: SkillCommand,
    ctx: MotionContext,
) -> Tuple[bool, str, float, PoseStamped]:
    """
    HANDOVER 스킬 실행:
      - perception에 "handover" pose 요청 (또는 cmd.target_pose 사용)
      - camera_link → base 좌표 변환 (ctx.transform_camera_to_base)
      - 접근/대기/오픈/리트랙트 수행
      - (success, message, confidence, final_pose) 반환
    """

    params_json = cmd.params_json

    # 1) target_pose가 이미 주어졌는지 확인
    if cmd.target_pose.header.frame_id:
        cam_pose = cmd.target_pose
        confidence = 1.0
        ctx.node.get_logger().info(
            f"[HANDOVER] 외부 제공 target_pose 사용 (frame_id={cam_pose.header.frame_id})"
        )
    else:
        # perception에 pose 요청 (object_name은 handover로 고정)
        pose_resp = ctx.request_object_pose("handover")
        if pose_resp is None:
            msg = "get_object_pose call failed"
            ctx.node.get_logger().error(f"[HANDOVER] {msg}")
            return False, msg, 0.0, PoseStamped()

        confidence = float(getattr(pose_resp, "confidence", 0.0))

        if not pose_resp.success:
            msg = f"get_object_pose 실패: {pose_resp.message}"
            ctx.node.get_logger().warn(f"[HANDOVER] {msg}")
            return False, msg, confidence, PoseStamped()

        if confidence < HANDOVER_CONF_TH:
            msg = f"conf={confidence:.2f} < HANDOVER_CONF_TH={HANDOVER_CONF_TH:.2f}, handover skip"
            ctx.node.get_logger().warn(f"[HANDOVER] {msg}")
            return False, msg, confidence, PoseStamped()

        cam_pose = pose_resp.pose

    # 2) params_json 파싱
    pre_x_offset = DEFAULT_PRE_X_OFFSET
    pre_z_offset = DEFAULT_PRE_Z_OFFSET
    approach_z_offset = DEFAULT_APPROACH_Z_OFFSET
    wait_sec = DEFAULT_WAIT_SEC

    if params_json:
        try:
            params = json.loads(params_json)
            ctx.node.get_logger().info(f"[HANDOVER] params_json = {params}")

            pre_x_offset = float(params.get("pre_x_offset", pre_x_offset))
            pre_z_offset = float(params.get("pre_z_offset", pre_z_offset))
            approach_z_offset = float(params.get("approach_z_offset", approach_z_offset))
            wait_sec = float(params.get("wait_sec", wait_sec))

        except json.JSONDecodeError:
            ctx.node.get_logger().warn(f"[HANDOVER] params_json 파싱 실패: {params_json}")

    # 3) camera_link → base 변환
    try:
        base_xyz = ctx.transform_camera_to_base(cam_pose)
        bx, by, bz = base_xyz
    except Exception as e:
        msg = f"transform_camera_to_base failed: {e}"
        ctx.node.get_logger().error(f"[HANDOVER] {msg}")
        return False, msg, confidence, PoseStamped()

    ctx.node.get_logger().info(
        f"[HANDOVER DEBUG] cam=({cam_pose.pose.position.x:.3f},"
        f"{cam_pose.pose.position.y:.3f},"
        f"{cam_pose.pose.position.z:.3f}) -> base=({bx:.3f},{by:.3f},{bz:.3f}), conf={confidence:.2f}"
    )

    # 4) 모션 수행
    try:
        execute_handover_motion(
            ctx,
            bx, by, bz,
            pre_x_offset=pre_x_offset,
            pre_z_offset=pre_z_offset,
            approach_z_offset=approach_z_offset,
            wait_sec=wait_sec,
        )
        success = True
        message = "OK"
    except Exception as e:
        success = False
        message = f"handover motion error: {e}"
        ctx.node.get_logger().error(f"❌ [HANDOVER] motion 중 예외: {e}")

    # 5) final_pose 구성 (handover 타겟 기준)
    final_pose = ctx.make_final_pose(bx, by, bz)
    return success, message, confidence, final_pose
