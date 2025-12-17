#!/usr/bin/env python3
"""
쓰레기 버리기 스킬 (DUMP)
지정된 위치로 이동 후 Z값을 -100 내린 후 그립을 해제합니다.
"""
import time
from dum_e_motion.motion_context import MotionContext

# 버리는 위치 좌표 (X, Y, Z, Rx, Ry, Rz)
DUMP_POSITION = [790.54, 177.5, 261.73, 19.66, 148.15, 11.76]
Z_OFFSET = -100.0  # Z값 오프셋


def run_dump_skill(ctx: MotionContext):
    """
    쓰레기 버리기 스킬 실행
    
    Args:
        ctx: MotionContext 객체
    
    Returns:
        (success, message, confidence)
    """
    try:
        ctx.logger.info('🗑️ 쓰레기 버리기 스킬 시작')
        
        ctx.logger.info('=' * 60)
        ctx.logger.info('🗑️ 쓰레기 버리기 시작')
        ctx.logger.info('=' * 60)
        
        # 버리는 위치로 이동
        dump_pos = DUMP_POSITION.copy()
        ctx.logger.info(f'버리는 위치로 이동 중...')
        ctx.logger.info(f'  X={dump_pos[0]:.2f}, Y={dump_pos[1]:.2f}, Z={dump_pos[2]:.2f}')
        
        if not ctx.move_line(dump_pos, vel=100.0, acc=120.0):
            ctx.logger.error('❌ 버리는 위치 이동 실패')
            return False, "Failed to move to dump position", 0.0
        
        ctx.logger.info('✅ 버리는 위치 도착')
        time.sleep(0.5)
        
        # Z값을 오프셋만큼 내림
        lower_pos = dump_pos.copy()
        lower_pos[2] += Z_OFFSET  # Z 내려가기
        
        ctx.logger.info(f'내려가는 중 (Z={lower_pos[2]:.2f})...')
        if not ctx.move_line(lower_pos, vel=50.0, acc=80.0):
            ctx.logger.error('❌ 내려가기 실패')
            return False, "Failed to move down", 0.0
        
        ctx.logger.info('✅ 내려갔습니다')
        time.sleep(0.3)
        
        # 그립 해제
        ctx.logger.info('🔓 그립 해제 중...')
        if not ctx.gripper.open():
            ctx.logger.error('❌ 그립 해제 실패')
            return False, "Failed to release gripper", 0.0
        
        ctx.logger.info('✅ 그립 해제 완료')
        time.sleep(0.5)
        
        # 원래 위치로 올라가기
        ctx.logger.info('올라가는 중...')
        if not ctx.move_line(dump_pos, vel=50.0, acc=80.0):
            ctx.logger.error('❌ 올라가기 실패')
            return False, "Failed to move up", 0.0
        
        ctx.logger.info('✅ 원래 위치로 복귀')
        
        ctx.logger.info('\n✅ 쓰레기 버리기 완료!')
        ctx.logger.info('=' * 60)
        
        return True, "Trash disposal completed successfully", 1.0
        
    except Exception as e:
        ctx.logger.error(f"❌ 쓰레기 버리기 실패: {e}")
        return False, f"Trash disposal failed: {e}", 0.0
