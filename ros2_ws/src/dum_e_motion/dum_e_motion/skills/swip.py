#!/usr/bin/env python3
"""
바닥 닦기 스킬 (SWIP)
정사각형 영역의 대각선 두 꼭지점을 기준으로 지그재그 패턴으로 청소합니다.
"""
import time
from dum_e_motion.motion_context import MotionContext

# 청소 영역 좌표 (X, Y, Z, Rx, Ry, Rz)
CORNER1 = [187.43, 217.71, 6.0, 107.52, 179.9, 107.59]  # 왼쪽 아래
CORNER2 = [575.96, -220.49, 10.91, 103.75, -178.28, 103.30]  # 오른쪽 위
STROKE_SPACING = 25.0


def run_swip_skill(ctx: MotionContext):
    """
    바닥 닦기 스킬 실행
    
    Args:
        ctx: MotionContext 객체
    
    Returns:
        (success, message, confidence)
    """
    try:
        ctx.logger.info('🧹 바닥 닦기 스킬 시작')
        
        # 영역 계산
        x_min = min(CORNER1[0], CORNER2[0])
        x_max = max(CORNER1[0], CORNER2[0])
        y_min = min(CORNER1[1], CORNER2[1])
        y_max = max(CORNER1[1], CORNER2[1])
        
        width = x_max - x_min
        height = y_max - y_min
        
        # Z, Rx, Ry, Rz는 corner1 기준으로 유지
        z = CORNER1[2]
        rx, ry, rz = CORNER1[3], CORNER1[4], CORNER1[5]
        
        ctx.logger.info('=' * 60)
        ctx.logger.info('🧹 영역 청소 시작')
        ctx.logger.info(f'  영역 크기: {width:.1f}mm × {height:.1f}mm')
        ctx.logger.info(f'  X 범위: {x_min:.1f} ~ {x_max:.1f}')
        ctx.logger.info(f'  Y 범위: {y_min:.1f} ~ {y_max:.1f}')
        ctx.logger.info(f'  지그재그 간격: {STROKE_SPACING}mm')
        ctx.logger.info('=' * 60)
        
        # Y 방향으로 몇 번 왕복할지 계산
        num_passes = int(height / STROKE_SPACING) + 1
        
        # 시작점으로 이동 (왼쪽 아래 모서리)
        start_pos = [x_min, y_min, z, rx, ry, rz]
        ctx.logger.info('시작점으로 이동 중...')
        
        if not ctx.move_line(start_pos, vel=100.0, acc=120.0):
            ctx.logger.error('❌ 시작점 이동 실패')
            return False, "Failed to move to start position", 0.0
        
        ctx.logger.info('✅ 시작점 도착')
        time.sleep(0.5)
        
        current_y = y_min
        go_right = True  # X+ 방향으로 갈지 여부
        
        for i in range(num_passes):
            ctx.logger.info(f'\n[Pass {i+1}/{num_passes}]')
            
            # 현재 Y 위치에서 X 방향으로 쭉 닦기
            if go_right:
                # 오른쪽으로 (X+)
                target_pos = [x_max, current_y, z, rx, ry, rz]
                ctx.logger.info(f'  → 오른쪽으로 닦기 (Y={current_y:.1f})...')
            else:
                # 왼쪽으로 (X-)
                target_pos = [x_min, current_y, z, rx, ry, rz]
                ctx.logger.info(f'  ← 왼쪽으로 닦기 (Y={current_y:.1f})...')
            
            if not ctx.move_line(target_pos, vel=80.0, acc=100.0):
                ctx.logger.error('❌ 이동 실패')
                return False, "Failed to move during wiping", 0.0
            
            ctx.logger.info('  ✅ 완료')
            time.sleep(0.2)
            
            # 다음 줄로 이동 (Y 방향)
            if i < num_passes - 1:
                current_y += STROKE_SPACING
                if current_y > y_max:
                    current_y = y_max
                
                if go_right:
                    next_pos = [x_max, current_y, z, rx, ry, rz]
                else:
                    next_pos = [x_min, current_y, z, rx, ry, rz]
                
                ctx.logger.info(f'  ↑ 다음 줄로 이동 (Y={current_y:.1f})...')
                if not ctx.move_line(next_pos, vel=60.0, acc=80.0):
                    ctx.logger.error('❌ 이동 실패')
                    return False, "Failed to move to next line", 0.0
                
                ctx.logger.info('  ✅ 완료')
                time.sleep(0.2)
                
                go_right = not go_right  # 방향 전환
        
        ctx.logger.info('\n✅ 바닥 닦기 완료!')
        ctx.logger.info('=' * 60)
        
        return True, "Floor wiping completed successfully", 1.0
        
    except Exception as e:
        ctx.logger.error(f"❌ 바닥 닦기 실패: {e}")
        return False, f"Floor wiping failed: {e}", 0.0
