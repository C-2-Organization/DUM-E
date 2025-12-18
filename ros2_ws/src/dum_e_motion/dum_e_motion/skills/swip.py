#!/usr/bin/env python3
"""
바닥 닦기 스킬 (SWIP)
정사각형 영역의 대각선 두 꼭지점을 기준으로 지그재그 패턴으로 청소합니다.
"""
from typing import Tuple
from geometry_msgs.msg import PoseStamped
from dum_e_motion.motion_context import MotionContext

# 청소 영역 좌표 (X, Y, Z, Rx, Ry, Rz)
CORNER1 = [187.43, 217.71, 6.0, 107.52, 179.9, 107.59]  # 왼쪽 아래
CORNER2 = [575.96, -220.49, 10.91, 103.75, -178.28, 103.30]  # 오른쪽 위
STROKE_SPACING = 25.0


def execute_swip_motion(ctx: MotionContext):
    """
    바닥 닦기 모션 실행
    """
    from DSR_ROBOT2 import DR_MV_MOD_ABS, DR_MV_RA_DUPLICATE, get_current_posx
    from DR_common2 import posx
    
    # 현재 로봇 위치의 회전값 가져오기
    current_pos = get_current_posx()[0]
    
    # 영역 계산
    x_min = min(CORNER1[0], CORNER2[0])
    x_max = max(CORNER1[0], CORNER2[0])
    y_min = min(CORNER1[1], CORNER2[1])
    y_max = max(CORNER1[1], CORNER2[1])
    
    width = x_max - x_min
    height = y_max - y_min
    
    # Z는 corner1 기준으로 유지, Rx, Ry, Rz는 현재 위치 기준
    z = CORNER1[2]
    rx, ry, rz = current_pos[3], current_pos[4], current_pos[5]
    
    ctx.node.get_logger().info('=' * 60)
    ctx.node.get_logger().info('🧹 영역 청소 시작')
    ctx.node.get_logger().info(f'  영역 크기: {width:.1f}mm × {height:.1f}mm')
    ctx.node.get_logger().info(f'  X 범위: {x_min:.1f} ~ {x_max:.1f}')
    ctx.node.get_logger().info(f'  Y 범위: {y_min:.1f} ~ {y_max:.1f}')
    ctx.node.get_logger().info(f'  지그재그 간격: {STROKE_SPACING}mm')
    ctx.node.get_logger().info('=' * 60)
    
    # Y 방향으로 몇 번 왕복할지 계산
    num_passes = int(height / STROKE_SPACING) + 1
    
    # 1) 시작점으로 이동 (왼쪽 아래 모서리)
    start_pos = posx([x_min, y_min, z, rx, ry, rz])
    ctx.node.get_logger().info('시작점으로 이동 중...')
    
    ctx.motion.movel(
        start_pos,
        vel=ctx.LIN_VEL,
        acc=ctx.LIN_ACC,
        mod=DR_MV_MOD_ABS,
        ra=DR_MV_RA_DUPLICATE,
    )
    
    ctx.node.get_logger().info('✅ 시작점 도착')
    ctx.motion.wait(0.5)
    
    current_y = y_min
    go_right = True  # X+ 방향으로 갈지 여부
    
    # 2) 지그재그 패턴으로 닦기
    for i in range(num_passes):
        ctx.node.get_logger().info(f'\n[Pass {i+1}/{num_passes}]')
        
        # 현재 Y 위치에서 X 방향으로 쭉 닦기
        if go_right:
            # 오른쪽으로 (X+)
            target_pos = posx([x_max, current_y, z, rx, ry, rz])
            ctx.node.get_logger().info(f'  → 오른쪽으로 닦기 (Y={current_y:.1f})...')
        else:
            # 왼쪽으로 (X-)
            target_pos = posx([x_min, current_y, z, rx, ry, rz])
            ctx.node.get_logger().info(f'  ← 왼쪽으로 닦기 (Y={current_y:.1f})...')
        
        ctx.motion.movel(
            target_pos,
            vel=[120.0, 200.0],
            acc=[100.0, 100.0],
            mod=DR_MV_MOD_ABS,
            ra=DR_MV_RA_DUPLICATE,
        )
        
        ctx.node.get_logger().info('  ✅ 완료')
        ctx.motion.wait(0.2)
        
        # 3) 다음 줄로 이동 (Y 방향)
        if i < num_passes - 1:
            current_y += STROKE_SPACING
            if current_y > y_max:
                current_y = y_max
            
            if go_right:
                next_pos = posx([x_max, current_y, z, rx, ry, rz])
            else:
                next_pos = posx([x_min, current_y, z, rx, ry, rz])
            
            ctx.node.get_logger().info(f'  ↑ 다음 줄로 이동 (Y={current_y:.1f})...')
            ctx.motion.movel(
                next_pos,
                vel=[100.0, 150.0],
                acc=[80.0, 80.0],
                mod=DR_MV_MOD_ABS,
                ra=DR_MV_RA_DUPLICATE,
            )
            
            ctx.node.get_logger().info('  ✅ 완료')
            ctx.motion.wait(0.2)
            
            go_right = not go_right  # 방향 전환
    
    ctx.node.get_logger().info('\n✅ 바닥 닦기 완료!')
    ctx.node.get_logger().info('=' * 60)


def run_swip_skill(ctx: MotionContext) -> Tuple[bool, str, float, PoseStamped]:
    """
    바닥 닦기 스킬 실행
    
    Args:
        ctx: MotionContext 객체
    
    Returns:
        (success, message, confidence, final_pose)
    """
    from DSR_ROBOT2 import get_current_posx
    
    try:
        ctx.node.get_logger().info('🧹 바닥 닦기 스킬 시작')
        execute_swip_motion(ctx)
        success = True
        message = "Floor wiping completed successfully"
        confidence = 1.0
    except Exception as e:
        success = False
        message = f"Floor wiping failed: {e}"
        confidence = 0.0
        ctx.node.get_logger().error(f"❌ 바닥 닦기 실패: {e}")
    
    current_pos = get_current_posx()[0]
    final_pose = ctx.make_final_pose(current_pos[0], current_pos[1], current_pos[2])
    return success, message, confidence, final_pose
