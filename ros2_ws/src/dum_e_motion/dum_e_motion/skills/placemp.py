# dum_e_motion/skills/placemp.py
import time
import json
from typing import Tuple

from geometry_msgs.msg import PoseStamped

from dum_e_interfaces.msg import SkillCommand
from dum_e_motion.motion_context import MotionContext

# 기존 place 모션 재사용
from dum_e_motion.skills.place import execute_place_motion

# ==============================================================================
# 설정 및 상수
# ==============================================================================
PLACEMP_CONF_TH = 0.2       # MediaPipe hand pose 신뢰도 기준
GRIPPER_OFFSET = 230        # 카메라 기준 검지손가락 → 그리퍼 TCP 보정 (mm)
MIN_SAFE_Z = 50             # base 기준 최소 z 높이 (mm)

PLACEMP_MAX_ATTEMPTS = 10    # MediaPipe 재시도 횟수
PLACEMP_RETRY_WAIT = 0.12   # 각 시도 사이 대기 (초) - 너무 길게 잡지 말자

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


def run_placemp_skill(
    cmd: SkillCommand,
    ctx: MotionContext,
) -> Tuple[bool, str, float, PoseStamped]:
    """
    PLACEMP 스킬 실행:

      - MediaPipe를 이용해 검지손가락(또는 손가락 끝) 위치를 인식한 포즈를 사용해서
        그 위치로 이동 → 물체 내려놓기.

      - 사용 방법:
        1) 외부에서 cmd.target_pose에 카메라 좌표계 포즈를 직접 넣어주거나
        2) perception 쪽에서 "placemp"에 해당하는 pose를 제공하도록 구현:
           pose_resp = ctx.request_object_pose("placemp")
    """

    ctx.node.get_logger().info("[PLACEMP] Start Skill (MediaPipe index finger)")

    # ------------------------------------------------------------------
    # 1) target_pose 우선 사용 여부
    # ------------------------------------------------------------------
    if _has_valid_external_pose(cmd.target_pose):
        cam_pose = cmd.target_pose
        confidence = 1.0
        ctx.node.get_logger().info(
            f"[PLACEMP] Using external target_pose (frame_id={cam_pose.header.frame_id})"
        )
    else:
        # ------------------------------------------------------------------
        # 2) Perception으로 MediaPipe 손가락 포즈 요청 (여러 번 재시도)
        # ------------------------------------------------------------------
        cam_pose = None
        confidence = 0.0
        last_msg = ""

        for attempt in range(1, PLACEMP_MAX_ATTEMPTS + 1):
            pose_resp = ctx.request_object_pose("placemp")
            if pose_resp is None:
                last_msg = "get_object_pose('placemp') call failed (None response)"
                ctx.node.get_logger().warn(
                    f"[PLACEMP] attempt {attempt}/{PLACEMP_MAX_ATTEMPTS}: {last_msg}"
                )
                confidence = 0.0
            else:
                confidence = float(getattr(pose_resp, "confidence", 0.0))

                if not pose_resp.success:
                    last_msg = f"get_object_pose('placemp') 실패: {pose_resp.message}"
                    ctx.node.get_logger().warn(
                        f"[PLACEMP] attempt {attempt}/{PLACEMP_MAX_ATTEMPTS}: {last_msg}"
                    )
                elif confidence < PLACEMP_CONF_TH:
                    last_msg = (
                        f"conf={confidence:.2f} < PLACEMP_CONF_TH={PLACEMP_CONF_TH:.2f}"
                    )
                    ctx.node.get_logger().warn(
                        f"[PLACEMP] attempt {attempt}/{PLACEMP_MAX_ATTEMPTS}: {last_msg}"
                    )
                else:
                    # ✅ 성공 케이스: 이 시점에서 MediaPipe가 손을 안정적으로 찾음
                    cam_pose = pose_resp.pose
                    ctx.node.get_logger().info(
                        f"[PLACEMP] MediaPipe hand detected on attempt "
                        f"{attempt}/{PLACEMP_MAX_ATTEMPTS}, conf={confidence:.2f}"
                    )
                    break  # 루프 탈출

            # 다음 프레임을 위해 잠깐 기다렸다가 재시도
            if attempt < PLACEMP_MAX_ATTEMPTS:
                time.sleep(PLACEMP_RETRY_WAIT)

        # 모든 시도 실패
        if cam_pose is None:
            msg = (
                f"PLACEMP: MediaPipe hand detection failed after "
                f"{PLACEMP_MAX_ATTEMPTS} attempts. last_msg='{last_msg}'"
            )
            ctx.node.get_logger().warn(f"[PLACEMP] {msg}")
            return False, msg, confidence, PoseStamped()


    # ------------------------------------------------------------------
    # 3) Camera 기준 검지손가락 포즈 → TCP 보정
    #    - cam_pose: 검지손가락 끝(or 근처) 위치 (camera_link 기준)
    #    - tcp_cam_pose: 그리퍼 TCP가 도달해야 할 포즈 (camera_link 기준)
    # ------------------------------------------------------------------
    tcp_cam_pose = PoseStamped()
    tcp_cam_pose.header = cam_pose.header
    tcp_cam_pose.pose.position.x = cam_pose.pose.position.x
    tcp_cam_pose.pose.position.y = cam_pose.pose.position.y
    tcp_cam_pose.pose.position.z = cam_pose.pose.position.z - GRIPPER_OFFSET
    tcp_cam_pose.pose.orientation = cam_pose.pose.orientation

    ctx.node.get_logger().info(
        "[PLACEMP] GRIPPER_OFFSET 적용: "
        f"cam_z={cam_pose.pose.position.z:.1f} -> "
        f"tcp_cam_z={tcp_cam_pose.pose.position.z:.1f}"
    )

    # ------------------------------------------------------------------
    # 4) camera_link → base 좌표 변환
    # ------------------------------------------------------------------
    try:
        bx, by, bz = ctx.transform_camera_to_base(tcp_cam_pose)
    except Exception as e:
        msg = f"transform_camera_to_base failed: {e}"
        ctx.node.get_logger().error(f"[PLACEMP] {msg}")
        return False, msg, confidence, PoseStamped()

    ctx.node.get_logger().info(
        f"[PLACEMP DEBUG] cam=({cam_pose.pose.position.x:.3f},"
        f"{cam_pose.pose.position.y:.3f},"
        f"{cam_pose.pose.position.z:.3f}) -> "
        f"base=({bx:.3f},{by:.3f},{bz:.3f}), conf={confidence:.2f}"
    )

    # ------------------------------------------------------------------
    # 5) 안전한 Z 높이 보정
    # ------------------------------------------------------------------
    if bz < MIN_SAFE_Z:
        ctx.node.get_logger().warn(
            f"[PLACEMP] Z={bz:.1f} too low, adjusting to {MIN_SAFE_Z}"
        )
        bz = MIN_SAFE_Z

    # ------------------------------------------------------------------
    # 6) 모션 실행 (PLACE 모션 재사용)
    #    execute_place_motion:
    #      - target로 이동 → open_gripper → 살짝 위로 → HOME
    # ------------------------------------------------------------------
    try:
        execute_place_motion(ctx, bx, by, bz)
        success = True
        message = "Success"
    except Exception as e:
        success = False
        message = f"PLACEMP motion error: {e}"
        ctx.node.get_logger().error(f"[PLACEMP] Motion Error: {e}")

    final_pose = ctx.make_final_pose(bx, by, bz)
    return success, message, confidence, final_pose
