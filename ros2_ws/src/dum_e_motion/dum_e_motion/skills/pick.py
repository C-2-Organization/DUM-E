# dum_e_motion/skills/pick.py
import json
from typing import Tuple

from geometry_msgs.msg import PoseStamped

from dum_e_interfaces.msg import SkillCommand
from dum_e_motion.motion_context import MotionContext


PICK_CONF_TH = 0.3  # perception에서 넘어온 confidence가 이보다 낮으면 pick 안 함
GRIPPER_OFFSET = 205

# ------------------------------------------------------------------
# Doosan + RG2 pick 모션
# ------------------------------------------------------------------
def execute_pick_motion(ctx: MotionContext, x, y, z):
    """
    접근 → 잡기 → 홈
    """
    from DSR_ROBOT2 import (
        DR_MV_MOD_ABS,
        DR_MV_RA_DUPLICATE,
        get_current_posx,
    )
    from DR_common2 import posx

    ctx.node.get_logger().info(
        f"[MOVE] Pick → base({x:.3f}, {y:.3f}, {z:.3f})"
    )

    current_pos = get_current_posx()[0]

    # 그리퍼 오픈
    ctx.motion.open_gripper()
    ctx.motion.wait(1)

    approach_pos = posx([
        x,
        y,
        z,
        current_pos[3],
        current_pos[4],
        current_pos[5],
    ])

    # 접근
    ctx.motion.movel(
        approach_pos,
        vel=ctx.LIN_VEL,
        acc=ctx.LIN_ACC,
        mod=DR_MV_MOD_ABS,
        ra=DR_MV_RA_DUPLICATE,
    )

    # 집기
    ctx.motion.close_gripper()
    ctx.motion.wait(1)

    # 홈으로
    ctx.motion.movej(
        ctx.CUSTOM_HOME_JOINT,
        vel=ctx.JNT_VEL,
        acc=ctx.JNT_ACC,
        mod=DR_MV_MOD_ABS,
        ra=DR_MV_RA_DUPLICATE,
    )

# ------------------------------------------------------------------
# Pick service call example
# ------------------------------------------------------------------
# ros2 service call /run_skill dum_e_interfaces/srv/RunSkill "{
#   command: {
#     skill_type: 0,
#     object_name: 'scissors',
#     target_pose: {
#       header: {frame_id: ''},
#       pose: {
#         position: {x: 0.0, y: 0.0, z: 0.0},
#         orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}
#       }
#     },
#     params_json: ''
#   }
# }"

def run_pick_skill(
    cmd: SkillCommand,
    ctx: MotionContext,
) -> Tuple[bool, str, float, PoseStamped]:
    """
    PICK 스킬 실행:
      - cmd.object_name / cmd.target_pose / cmd.params_json 사용
      - 필요 시 perception에 pose 요청
      - camera_link → base 좌표 변환
      - Doosan + RG2로 pick 모션 실행
      - (success, message, confidence, final_pose) 반환
    """
    object_name = cmd.object_name.strip()
    params_json = cmd.params_json

    if not object_name:
        msg = "object_name is empty"
        ctx.node.get_logger().warn(f"[PICK] {msg}")
        return False, msg, 0.0, PoseStamped()

    ctx.node.get_logger().info(
        f"[PICK] skill 실행: object_name='{object_name}'"
    )

    # 1) target_pose가 이미 주어졌는지 확인 (frame_id가 비어있지 않으면 사용)
    if cmd.target_pose.header.frame_id:
        cam_pose = cmd.target_pose
        confidence = 1.0  # 외부에서 pose를 신뢰한다고 가정
        ctx.node.get_logger().info(
            f"[PICK] 외부 제공 target_pose 사용 (frame_id={cam_pose.header.frame_id})"
        )
    else:
        # perception에 pose 요청
        pose_resp = ctx.request_object_pose(object_name)
        if pose_resp is None:
            msg = "get_object_pose call failed"
            ctx.node.get_logger().error(msg)
            return False, msg, 0.0, PoseStamped()

        confidence = float(pose_resp.confidence)

        if not pose_resp.success:
            msg = f"get_object_pose 실패: {pose_resp.message}"
            ctx.node.get_logger().warn(msg)
            return False, msg, confidence, PoseStamped()

        if confidence < PICK_CONF_TH:
            msg = (
                f"conf={confidence:.2f} < PICK_CONF_TH={PICK_CONF_TH:.2f}, pick skip"
            )
            ctx.node.get_logger().warn(msg)
            return False, msg, confidence, PoseStamped()

        cam_pose = pose_resp.pose

    # params_json 파싱 (나중에 tilt, offset 등에 활용 가능)
    if params_json:
        try:
            params = json.loads(params_json)
            ctx.node.get_logger().info(f"[PICK] params_json = {params}")
        except json.JSONDecodeError:
            ctx.node.get_logger().warn(
                f"[PICK] params_json 파싱 실패: {params_json}"
            )

    # 2) camera_link → base 좌표 변환
    # cam_pose는 "물체 중심 좌표" (카메라 기준)
    # TCP(그리퍼 베이스)는 물체보다 카메라 쪽으로 205mm 뒤에 있어야 하므로,
    # 카메라 프레임에서 z축으로 0.205m 빼준 위치를 TCP 목표로 사용.
    tcp_cam_pose = PoseStamped()
    tcp_cam_pose.header = cam_pose.header  # frame_id='camera_link' 유지
    tcp_cam_pose.pose = cam_pose.pose

    # RealSense 기준: +Z가 카메라 앞 방향이라고 가정.
    # 그리퍼가 카메라 앞쪽으로 205mm 나와 있으니,
    # TCP는 물체보다 카메라 쪽으로 0.205m 뒤에 있어야 한다 → z -= 0.205
    tcp_cam_pose.pose.position.z -= GRIPPER_OFFSET

    # 혹시 depth가 0.205m보다 작은 비정상적인 경우 방어
    if tcp_cam_pose.pose.position.z <= 0.0:
        msg = (
            f"computed tcp_cam_pose.z={tcp_cam_pose.pose.position.z:.3f} <= 0.0, "
            f"invalid for GRIPPER_OFFSET={GRIPPER_OFFSET}"
        )
        ctx.node.get_logger().warn(f"[PICK] {msg}")
        return False, msg, confidence, PoseStamped()

    base_xyz = ctx.transform_camera_to_base(tcp_cam_pose)
    bx, by, bz = base_xyz

    ctx.node.get_logger().info(
        f"[PICK DEBUG] target='{object_name}', "
        f"cam_obj=({cam_pose.pose.position.x:.3f},"
        f"{cam_pose.pose.position.y:.3f},"
        f"{cam_pose.pose.position.z:.3f}), "
        f"tcp_cam=({tcp_cam_pose.pose.position.x:.3f},"
        f"{tcp_cam_pose.pose.position.y:.3f},"
        f"{tcp_cam_pose.pose.position.z:.3f}), "
        f"base_tcp=({bx:.3f},{by:.3f},{bz:.3f}), "
        f"conf={confidence:.2f}"
    )

    # 3) 실제 모션 수행
    try:
        execute_pick_motion(ctx, bx, by, bz)
        success = True
        message = "OK"
    except Exception as e:
        success = False
        message = f"pick motion error: {e}"
        ctx.node.get_logger().error(f"❌ pick motion 중 예외: {e}")

    # 4) final_pose 구성
    final_pose = ctx.make_final_pose(bx, by, bz)
    return success, message, confidence, final_pose
