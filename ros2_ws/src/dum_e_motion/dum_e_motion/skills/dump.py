#!/usr/bin/env python3
"""
쓰레기 버리기 스킬 (DUMP)
지정된 위치로 이동 후 Z값을 -100 내린 후 그립을 해제합니다.
"""
from typing import Tuple
from geometry_msgs.msg import PoseStamped
from dum_e_motion.motion_context import MotionContext

# 버리는 위치 좌표 (X, Y, Z, Rx, Ry, Rz)
DUMP_POSITION = [790.54, 177.5, 261.73, 19.66, 148.15, 11.76]
Z_OFFSET = -100.0  # Z값 오프셋


def execute_dump_motion(ctx: MotionContext):
    """
    쓰레기 버리기 모션 실행
    """
    from DSR_ROBOT2 import (
        posx,
        DR_MV_MOD_ABS,
        DR_MV_MOD_REL,
        DR_MV_RA_DUPLICATE,
    )

    dump_x, dump_y, dump_z, _, _, _ = DUMP_POSITION
    dump_pos = posx(DUMP_POSITION)

    ctx.node.get_logger().info('=' * 60)
    ctx.node.get_logger().info('🗑️ 쓰레기 버리기 시작')
    ctx.node.get_logger().info('=' * 60)

    # 1) 버리는 위치로 이동

    ctx.node.get_logger().info(f'버리는 위치로 이동 중...')
    ctx.node.get_logger().info(f'  X={dump_x:.2f}, Y={dump_y:.2f}, Z={dump_z:.2f}')

    ctx.motion.movel(
        dump_pos,
        vel=ctx.LIN_VEL,
        acc=ctx.LIN_ACC,
        mod=DR_MV_MOD_ABS,
        ra=DR_MV_RA_DUPLICATE,
    )

    ctx.node.get_logger().info('✅ 버리는 위치 도착')
    ctx.motion.wait(0.5)

    # 2) Z값을 오프셋만큼 내림

    down_pos = posx(0, 0, Z_OFFSET, 0, 0, 0)

    # ctx.node.get_logger().info(f'내려가는 중 (Z={Z_OFFSET:.2f})...')
    ctx.motion.movel(
        down_pos,
        vel=ctx.LIN_VEL,
        acc=ctx.LIN_ACC,
        mod=DR_MV_MOD_REL,
        ra=DR_MV_RA_DUPLICATE,
    )

    ctx.node.get_logger().info('✅ 내려갔습니다')
    ctx.motion.wait(0.3)

    # 3) 그립 해제
    ctx.node.get_logger().info('🔓 그립 해제 중...')
    ctx.motion.open_gripper()

    ctx.node.get_logger().info('✅ 그립 해제 완료')
    ctx.motion.wait(0.5)

    # 4) 원래 위치로 올라가기
    ctx.node.get_logger().info('올라가는 중...')
    ctx.motion.movel(
        dump_pos,
        vel=[100.0, 150.0],
        acc=[100.0, 100.0],
        mod=DR_MV_MOD_ABS,
        ra=DR_MV_RA_DUPLICATE,
    )

    ctx.node.get_logger().info('✅ 원래 위치로 복귀')
    ctx.node.get_logger().info('\n✅ 쓰레기 버리기 완료!')
    ctx.node.get_logger().info('=' * 60)


def run_dump_skill(ctx: MotionContext) -> Tuple[bool, str, float, PoseStamped]:
    """
    쓰레기 버리기 스킬 실행

    Args:
        ctx: MotionContext 객체

    Returns:
        (success, message, confidence, final_pose)
    """
    from DSR_ROBOT2 import get_current_posx

    try:
        ctx.node.get_logger().info('🗑️ 쓰레기 버리기 스킬 시작')
        execute_dump_motion(ctx)
        success = True
        message = "Trash disposal completed successfully"
        confidence = 1.0
    except Exception as e:
        success = False
        message = f"Trash disposal failed: {e}"
        confidence = 0.0
        ctx.node.get_logger().error(f"❌ 쓰레기 버리기 실패: {e}")

    current_pos = get_current_posx()[0]
    final_pose = ctx.make_final_pose(current_pos[0], current_pos[1], current_pos[2])
    return success, message, confidence, final_pose
