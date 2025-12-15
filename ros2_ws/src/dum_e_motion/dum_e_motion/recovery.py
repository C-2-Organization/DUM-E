#!/usr/bin/env python3
"""
충돌 복구 모듈 (Collision Recovery)

doosan_rokey_collabo1 프로젝트 기반 충돌 복구 시스템:

복구 시퀀스:
1. SAFE_STOP 리셋 (set_robot_control: 2)
2. RECOVERY 모드 진입 (set_safety_mode: mode=2, event=0)
3. Z축 Jog 상승 (바닥 충돌 시, 100mm 기준)
4. RECOVERY 완료 (set_safety_mode: mode=2, event=2)
5. RECOVERY 모드 해제 (set_robot_control: 7)
6. 서보 ON (set_robot_control: 3)
7. 그리퍼 상태 확인 후 적절한 복구 동작 실행

복구 시나리오:
- 그립 상태 (물체 잡고 있음): 컨베이어/테이블로 이동 → Place → 홈
- 비그립 상태: 홈으로 직행
"""

import time
import threading
import subprocess
import signal
import os
import sys
from typing import Callable, Optional
from enum import IntEnum
from datetime import datetime

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup

try:
    from dsr_msgs2.srv import SetRobotControl, SetSafetyMode, Jog, GetCurrentPosx, GetRobotState
    DSR_AVAILABLE = True
except ImportError:
    DSR_AVAILABLE = False
    print("[Recovery] ⚠️ dsr_msgs2 없음 - 시뮬레이션 모드")


# =========================================
# 상수 정의
# =========================================
class RobotState(IntEnum):
    """로봇 상태 코드"""
    INITIALIZING = 0
    STANDBY = 1
    MOVING = 2
    SAFE_OFF = 3
    TEACHING = 4
    SAFE_STOP = 5
    EMERGENCY_STOP = 6
    HOMMING = 7
    RECOVERY = 8
    SAFE_STOP2 = 9
    SAFE_OFF2 = 10


class ControlCode(IntEnum):
    """제어 명령 코드"""
    RESET_SAFE_STOP = 2  # SAFE_STOP 리셋
    SERVO_ON = 3          # 서보 ON (SAFE_OFF 리셋)
    RESET_RECOVERY = 7    # RECOVERY 모드 해제


class SafetyMode(IntEnum):
    """안전 모드 설정"""
    RECOVERY = 2


class SafetyEvent(IntEnum):
    """안전 이벤트"""
    ENTER = 0      # 진입
    COMPLETE = 2   # 완료


# 복구 설정
RECOVERY_Z_THRESHOLD = 100.0    # 바닥 충돌 판별 기준 Z 높이 (mm)
RECOVERY_JOG_TIME = 1.5         # Jog 상승 시간 (초)
RECOVERY_JOG_SPEED = 20.0       # Jog 상승 속도 (mm/s)
RECOVERY_JOG_AXIS_Z = 2         # Z축 (Task 좌표계)

# 이동 속도
VELOCITY_MOVE = 30.0   # mm/s
ACCEL_MOVE = 60.0      # mm/s²

# 기본 위치 (실제 시스템에 맞게 수정 필요)
HOME_POSITION = [367.69, 7.38, 425.09, 83.88, 179.96, 83.73]  # [x, y, z, rx, ry, rz]
SAFE_DROP_POSITION = [300.0, 0.0, 300.0, 83.88, 179.96, 83.73]  # 물체 내려놓을 안전 위치


def state_name(state_code: int) -> str:
    """상태 코드를 이름으로 변환"""
    try:
        return RobotState(state_code).name
    except ValueError:
        return f"UNKNOWN({state_code})"


class CollisionRecovery:
    """
    충돌 복구 클래스
    
    충돌 감지 시 자동으로 6단계 복구 시퀀스를 수행하고,
    그리퍼 상태에 따라 적절한 후속 동작(Place/Home)을 실행합니다.
    """
    
    def __init__(
        self,
        node: Node,
        robot_controller=None,
        callback_group: Optional[ReentrantCallbackGroup] = None
    ):
        """
        Args:
            node: ROS2 노드
            robot_controller: 로봇 제어 인스턴스 (movel, get_current_posx, gripper 등)
            callback_group: 서비스 호출용 콜백 그룹
        """
        self.node = node
        self.robot = robot_controller
        self.callback_group = callback_group or ReentrantCallbackGroup()
        
        # 복구 상태
        self._is_recovering = False
        self._recovery_caused_by_collision = False
        self._saved_work_state = None
        
        # movel 실행 추적 (강제 중단 가능)
        self._movel_in_progress = False
        self._movel_stop_requested = False
        
        # 콜백
        self._on_progress: Optional[Callable[[str, int], None]] = None
        self._on_complete: Optional[Callable[[bool, bool], None]] = None
        
        # 로그 파일
        self.log_file = '/tmp/recovery_monitor.log'
        
        # 서비스 클라이언트 초기화
        self._init_clients()
        
        self.node.get_logger().info('[Recovery] 초기화 완료')
        self._write_log('info', '[Recovery] 초기화 완료')
    
    def _write_log(self, level: str, message: str):
        """모니터링 로그 파일에 기록"""
        try:
            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            log_line = f"[{timestamp}] [{level.upper()}] {message}\n"
            with open(self.log_file, 'a') as f:
                f.write(log_line)
        except Exception as e:
            self.node.get_logger().debug(f'[Recovery] 로그 파일 쓰기 실패: {e}')
    
    def _init_clients(self):
        """ROS2 서비스 클라이언트 초기화"""
        if not DSR_AVAILABLE:
            self.cli_control = None
            self.cli_safety = None
            self.cli_jog = None
            self.cli_posx = None
            return
        
        robot_id = 'dsr01'  # 환경에 맞게 수정
        
        self.cli_control = self.node.create_client(
            SetRobotControl,
            f'/{robot_id}/system/set_robot_control',
            callback_group=self.callback_group
        )
        
        self.cli_safety = self.node.create_client(
            SetSafetyMode,
            f'/{robot_id}/system/set_safety_mode',
            callback_group=self.callback_group
        )
        
        self.cli_jog = self.node.create_client(
            Jog,
            f'/{robot_id}/motion/jog',
            callback_group=self.callback_group
        )
        
        self.cli_posx = self.node.create_client(
            GetCurrentPosx,
            f'/{robot_id}/aux_control/get_current_posx',
            callback_group=self.callback_group
        )
        
        self.cli_state = self.node.create_client(
            GetRobotState,
            f'/{robot_id}/system/get_robot_state',
            callback_group=self.callback_group
        )
    
    # =========================================
    # 콜백 설정
    # =========================================
    def set_progress_callback(self, callback: Callable[[str, int], None]):
        """복구 진행 상태 콜백 설정"""
        self._on_progress = callback
    
    def set_complete_callback(self, callback: Callable[[bool, bool], None]):
        """복구 완료 콜백 설정 (success, was_gripping)"""
        self._on_complete = callback
    
    def _notify_progress(self, message: str, percent: int):
        """진행 상태 알림"""
        self.node.get_logger().info(f'[Recovery] {message} ({percent}%)')
        if self._on_progress:
            try:
                self._on_progress(message, percent)
            except Exception as e:
                self.node.get_logger().error(f'진행 콜백 에러: {e}')
    
    # =========================================
    # 서비스 호출
    # =========================================
    def _call_control(self, control_code: int) -> bool:
        """SetRobotControl 서비스 호출"""
        if not DSR_AVAILABLE or self.cli_control is None:
            self.node.get_logger().warn('[Recovery] SetRobotControl 서비스 없음 (시뮬레이션)')
            return True
        
        if not self.cli_control.wait_for_service(timeout_sec=1.0):
            self.node.get_logger().warn('[Recovery] SetRobotControl 서비스 대기 실패')
            return False
        
        req = SetRobotControl.Request()
        req.robot_control = control_code
        
        self.node.get_logger().info(f'[Recovery Debug] SetRobotControl 호출: code={control_code}')
        self._write_log('info', f'SetRobotControl 호출: code={control_code}')
        
        future = self.cli_control.call_async(req)
        start = time.time()
        # 타임아웃을 1초로 단축 (SAFE_STOP 중 응답 지연 방지)
        while not future.done() and (time.time() - start) < 1.0:
            time.sleep(0.05)
        
        if future.done() and future.result():
            result = future.result()
            self.node.get_logger().info(f'[Recovery Debug] SetRobotControl 결과: success={result.success}')
            self._write_log('info', f'SetRobotControl 결과: success={result.success}')
            return result.success
        
        # 타임아웃 시에도 로그만 남기고 계속 진행
        self.node.get_logger().warn('[Recovery Debug] SetRobotControl 타임아웃 (계속 진행)')
        self._write_log('warn', 'SetRobotControl 타임아웃')
        return True  # 타임아웃해도 계속 진행
    
    def _call_safety(self, mode: int, event: int) -> bool:
        """SetSafetyMode 서비스 호출"""
        if not DSR_AVAILABLE or self.cli_safety is None:
            self.node.get_logger().warn('[Recovery] SetSafetyMode 서비스 없음 (시뮬레이션)')
            return True
        
        if not self.cli_safety.wait_for_service(timeout_sec=1.0):
            self.node.get_logger().warn('[Recovery] SetSafetyMode 서비스 대기 실패')
            return False
        
        req = SetSafetyMode.Request()
        req.safety_mode = mode
        req.safety_event = event
        
        self.node.get_logger().info(f'[Recovery Debug] SetSafetyMode 호출: mode={mode}, event={event}')
        self._write_log('info', f'SetSafetyMode 호출: mode={mode}, event={event}')
        
        future = self.cli_safety.call_async(req)
        start = time.time()
        # 타임아웃을 1초로 단축
        while not future.done() and (time.time() - start) < 1.0:
            time.sleep(0.05)
        
        if future.done() and future.result():
            result = future.result()
            self.node.get_logger().info(f'[Recovery Debug] SetSafetyMode 결과: success={result.success}')
            self._write_log('info', f'SetSafetyMode 결과: success={result.success}')
            return result.success
        
        # 타임아웃 시에도 계속 진행
        self.node.get_logger().warn('[Recovery Debug] SetSafetyMode 타임아웃 (계속 진행)')
        self._write_log('warn', 'SetSafetyMode 타임아웃')
        return True  # 타임아웃해도 계속 진행
    
    def _call_jog(self, axis: int, speed: float, duration: float) -> bool:
        """Jog 서비스 호출 (시작 → 대기 → 정지)"""
        if not DSR_AVAILABLE or self.cli_jog is None:
            self.node.get_logger().warn('[Recovery] Jog 서비스 없음 (시뮬레이션)')
            return True
        
        if not self.cli_jog.wait_for_service(timeout_sec=1.0):
            self.node.get_logger().warn('[Recovery] Jog 서비스 없음')
            return False
        
        # Jog 시작
        req = Jog.Request()
        req.jog_axis = axis
        req.move_reference = 0  # BASE
        req.speed = speed
        
        future = self.cli_jog.call_async(req)
        start = time.time()
        while not future.done() and (time.time() - start) < 2.0:
            time.sleep(0.05)
        
        if not (future.done() and future.result() and future.result().success):
            return False
        
        # 대기
        time.sleep(duration)
        
        # Jog 정지
        req.speed = 0.0
        future = self.cli_jog.call_async(req)
        start = time.time()
        while not future.done() and (time.time() - start) < 2.0:
            time.sleep(0.05)
        
        return future.done() and future.result() and future.result().success
    
    def _safe_movel(self, position, vel: float = VELOCITY_MOVE, acc: float = ACCEL_MOVE, timeout: float = 30.0) -> bool:
        """
        안전한 movel 호출 (SAFE_STOP 감지 시 강제 중단)
        
        Args:
            position: 목표 위치
            vel: 속도 (mm/s)
            acc: 가속도 (mm/s²)
            timeout: 타임아웃 (초)
        
        Returns:
            성공 여부
        """
        if not self.robot or not hasattr(self.robot, 'movel'):
            self.node.get_logger().warn('[Recovery] robot.movel 메서드 없음')
            return True
        
        # movel 실행 추적
        self._movel_in_progress = True
        self._movel_stop_requested = False
        
        try:
            # 별도 스레드에서 movel 실행 (중단 가능하게)
            result = {'success': False}
            
            def _movel_thread():
                try:
                    success = self.robot.movel(position, vel=vel, acc=acc)
                    result['success'] = success
                except Exception as e:
                    self.node.get_logger().error(f'[Recovery] movel 예외: {e}')
                    result['success'] = False
            
            thread = threading.Thread(target=_movel_thread, daemon=True)
            thread.start()
            
            # 타임아웃 동안 대기 (주기적으로 중단 요청 확인)
            start = time.time()
            while thread.is_alive() and (time.time() - start) < timeout:
                # SAFE_STOP 감지 시 중단 요청
                if self._movel_stop_requested:
                    self.node.get_logger().warn('[Recovery] ⚠️ SAFE_STOP 감지 - movel 중단!')
                    self._write_log('warn', 'SAFE_STOP 감지 - movel 중단')
                    # 주의: 스레드는 daemon=True이므로 자동 정리됨
                    return False
                time.sleep(0.1)
            
            # 타임아웃 체크
            if thread.is_alive():
                self.node.get_logger().warn(f'[Recovery] ⚠️ movel 타임아웃 ({timeout}초)')
                self._write_log('warn', f'movel 타임아웃 ({timeout}초)')
                self._movel_stop_requested = True
                return False
            
            return result.get('success', False)
        
        finally:
            self._movel_in_progress = False
            self._movel_stop_requested = False
    
    def request_movel_stop(self):
        """movel 중단 요청 (SAFE_STOP 감지 시 호출)"""
        if self._movel_in_progress:
            self._movel_stop_requested = True
            self.node.get_logger().warn('[Recovery] movel 중단 요청 플래그 설정')
            self._write_log('warn', 'movel 중단 요청')
    
    def _get_current_z(self) -> Optional[float]:
        """현재 Z 좌표 조회 (GetCurrentPosx 서비스)"""
        if not DSR_AVAILABLE or self.cli_posx is None:
            self.node.get_logger().warn('[Recovery] GetCurrentPosx 서비스 없음 (시뮬레이션)')
            return None
        
        try:
            # 서비스 대기
            if not self.cli_posx.wait_for_service(timeout_sec=2.0):
                self.node.get_logger().warn('[Recovery] GetCurrentPosx 서비스 타임아웃')
                return None
            
            # 요청 생성
            req = GetCurrentPosx.Request()
            req.ref = 0  # BASE 좌표계
            
            # 서비스 호출
            future = self.cli_posx.call_async(req)
            start = time.time()
            while not future.done() and (time.time() - start) < 3.0:
                time.sleep(0.05)
            
            if not future.done():
                self.node.get_logger().warn('[Recovery] GetCurrentPosx 호출 타임아웃')
                return None
            
            result = future.result()
            if result and result.success and result.task_pos_info:
                # task_pos_info[0].data에서 Z (index 2) 추출
                pos_data = result.task_pos_info[0].data
                if pos_data and len(pos_data) >= 3:
                    z_value = float(pos_data[2])
                    self.node.get_logger().debug(f'[Recovery] 현재 Z: {z_value:.1f}mm')
                    return z_value
            
            self.node.get_logger().warn(f'[Recovery] GetCurrentPosx 실패: success={result.success if result else None}')
            return None
            
        except Exception as e:
            self.node.get_logger().error(f'[Recovery] Z 좌표 조회 예외: {e}')
            return None
    
    # =========================================
    # 복구 단계
    # =========================================
    def reset_safe_stop(self) -> bool:
        """
        1단계: SAFE_STOP 리셋
        SetRobotControl(2) 호출 후 상태 확인
        리셋 후에도 SAFE_STOP이면 빠르게 재시도
        """
        self._notify_progress('SAFE_STOP 리셋', 10)
        self._write_log('info', '[Recovery] 단계 1: SAFE_STOP 리셋 시작')
        
        for retry in range(3):  # 최대 3회 재시도
            self.node.get_logger().warn(f'[Recovery] SAFE_STOP 리셋 시도 {retry + 1}/3')
            
            result = self._call_control(ControlCode.RESET_SAFE_STOP)
            # 재시도 사이 대기를 0.3초로 단축
            time.sleep(0.3)
            
            # 상태 확인 (GetRobotState 호출) - 빠르게 확인
            if self.robot or True:  # 항상 상태 확인
                state = self._get_robot_state_fast()  # 빠른 버전
                state_str = state_name(state) if state else 'UNKNOWN'
                
                if state == RobotState.STANDBY:
                    self.node.get_logger().info(f'[Recovery] ✅ SAFE_STOP 리셋 성공 - 상태: {state_str}')
                    self._write_log('info', f'✅ SAFE_STOP 리셋 성공 - 상태: {state_str}')
                    return True
                elif state == RobotState.SAFE_STOP or state == RobotState.SAFE_STOP2:
                    self.node.get_logger().warn(f'[Recovery] ⚠️ 여전히 SAFE_STOP (시도 {retry+1}/3) - 상태: {state_str}')
                    self._write_log('warn', f'⚠️ 여전히 SAFE_STOP (시도 {retry+1}/3)')
                    # 더 짧은 대기로 빠르게 재시도
                    time.sleep(0.5)
                    continue
                else:
                    self.node.get_logger().info(f'[Recovery] 리셋 후 상태: {state_str}')
                    self._write_log('info', f'리셋 후 상태: {state_str}')
                    return result
            
            if result:
                return True
        
        self.node.get_logger().error('[Recovery] ❌ SAFE_STOP 리셋 실패 (3회 재시도 후)')
        self._write_log('error', '❌ SAFE_STOP 리셋 실패')
        return False
    
    def _get_robot_state_fast(self) -> Optional[int]:
        """빠른 로봇 상태 조회 (타임아웃 단축)"""
        if not DSR_AVAILABLE or self.cli_state is None:
            return None
        
        try:
            if not self.cli_state.wait_for_service(timeout_sec=0.5):
                return None
            
            req = GetRobotState.Request()
            future = self.cli_state.call_async(req)
            
            start = time.time()
            # 타임아웃을 0.5초로 단축
            while not future.done() and (time.time() - start) < 0.5:
                time.sleep(0.05)
            
            if future.done():
                result = future.result()
                if result:
                    return result.robot_state
            return None
            
        except Exception as e:
            self.node.get_logger().debug(f'[Recovery] get_robot_state_fast 예외: {e}')
            return None
    
    def _get_robot_state(self) -> Optional[int]:
        """현재 로봇 상태 조회 (GetRobotState 서비스)"""
        if not DSR_AVAILABLE or self.cli_state is None:
            self.node.get_logger().warn('[Recovery] GetRobotState 서비스 없음')
            return None
        
        try:
            # 서비스 대기
            if not self.cli_state.wait_for_service(timeout_sec=2.0):
                self.node.get_logger().warn('[Recovery] GetRobotState 서비스 타임아웃')
                return None
            
            # 요청 생성 및 호출
            req = GetRobotState.Request()
            future = self.cli_state.call_async(req)
            
            start = time.time()
            while not future.done() and (time.time() - start) < 2.0:
                time.sleep(0.05)
            
            if not future.done():
                self.node.get_logger().warn('[Recovery] GetRobotState 호출 타임아웃')
                return None
            
            result = future.result()
            if result:
                return result.robot_state
            return None
            
        except Exception as e:
            self.node.get_logger().error(f'[Recovery] get_robot_state 예외: {e}')
            return None
    
    def enter_recovery(self) -> bool:
        """2단계: RECOVERY 모드 진입"""
        self._notify_progress('복구 모드 진입', 25)
        self._write_log('info', '[Recovery] 단계 2: RECOVERY 모드 진입')
        result = self._call_safety(SafetyMode.RECOVERY, SafetyEvent.ENTER)
        time.sleep(0.3)
        if result:
            self._write_log('info', '✅ RECOVERY 모드 진입 성공')
        else:
            self._write_log('error', '❌ RECOVERY 모드 진입 실패')
        return result
    
    def jog_up(self) -> bool:
        """3단계: Z축 상승 (바닥 충돌 시)"""
        self._notify_progress('Z축 상승', 50)
        self._write_log('info', f'[Recovery] 단계 3: Z축 상승 ({RECOVERY_JOG_SPEED}mm/s, {RECOVERY_JOG_TIME}초)')
        result = self._call_jog(RECOVERY_JOG_AXIS_Z, RECOVERY_JOG_SPEED, RECOVERY_JOG_TIME)
        time.sleep(0.3)
        if result:
            self._write_log('info', '✅ Z축 상승 완료')
        else:
            self._write_log('error', '❌ Z축 상승 실패')
        return result
    
    def complete_recovery(self) -> bool:
        """4단계: RECOVERY 완료"""
        self._notify_progress('복구 완료 처리', 70)
        self._write_log('info', '[Recovery] 단계 4: RECOVERY 완료 처리')
        result = self._call_safety(SafetyMode.RECOVERY, SafetyEvent.COMPLETE)
        time.sleep(0.5)
        if result:
            self._write_log('info', '✅ RECOVERY 완료 처리 성공')
        else:
            self._write_log('error', '❌ RECOVERY 완료 처리 실패')
        return result
    
    def exit_recovery(self) -> bool:
        """5단계: RECOVERY 모드 해제"""
        self._notify_progress('복구 모드 종료', 85)
        self._write_log('info', '[Recovery] 단계 5: RECOVERY 모드 해제')
        result = self._call_control(ControlCode.RESET_RECOVERY)
        time.sleep(0.5)
        if result:
            self._write_log('info', '✅ RECOVERY 모드 해제 성공')
        else:
            self._write_log('error', '❌ RECOVERY 모드 해제 실패')
        return result
    
    def servo_on(self) -> bool:
        """6단계: 서보 ON"""
        self._notify_progress('서보 ON', 95)
        self._write_log('info', '[Recovery] 단계 6: 서보 ON')
        self.node.get_logger().info('[Recovery] 🔧 서보 ON 시작...')
        
        result = self._call_control(ControlCode.SERVO_ON)
        time.sleep(1.0)
        
        if result:
            self._write_log('info', '✅ 서보 ON 완료')
            self.node.get_logger().info('[Recovery] ✅ 서보 ON 완료')
            
            # 🔴 서보 ON 후 상태 확인
            time.sleep(0.5)
            current_state = self._get_robot_state()
            if current_state is not None:
                from .recovery import state_name
                self.node.get_logger().info(f'[Recovery] 📊 서보 ON 후 상태: {state_name(current_state)}')
                if current_state == RobotState.SAFE_STOP or current_state == RobotState.SAFE_OFF:
                    self.node.get_logger().warn(f'[Recovery] ⚠️ 서보 ON 후에도 SAFE 상태: {state_name(current_state)}')
        else:
            self._write_log('error', '❌ 서보 ON 실패')
            self.node.get_logger().error('[Recovery] ❌ 서보 ON 실패')
        
        return result
    
    # =========================================
    # 복구 후 동작
    # =========================================
    def _move_to_home(self) -> bool:
        """홈 위치로 이동"""
        if self.robot is None:
            self.node.get_logger().warn('[Recovery] robot_controller 없음')
            return False
        
        try:
            self.node.get_logger().info('[Recovery] 홈 위치로 이동...')
            
            if hasattr(self.robot, 'movel'):
                success = self._safe_movel(HOME_POSITION, vel=VELOCITY_MOVE, acc=ACCEL_MOVE)
                if success:
                    self.node.get_logger().info('[Recovery] ✅ 홈 도착')
                else:
                    self.node.get_logger().warn('[Recovery] ⚠️ 홈 이동 실패')
                return success
            else:
                self.node.get_logger().warn('[Recovery] robot.movel 메서드 없음')
                return False
                
        except Exception as e:
            self.node.get_logger().error(f'[Recovery] 홈 이동 예외: {e}')
            return False
    
    def _place_and_go_home(self) -> bool:
        """그립 상태에서 복구: 안전 위치에 물체 내려놓고 홈으로 이동"""
        if self.robot is None:
            self.node.get_logger().warn('[Recovery] robot_controller 없음')
            return False
        
        try:
            self.node.get_logger().info('[Recovery] 📦 물체 반납 시퀀스 시작')
            
            # 1. 현재 위치에서 안전 높이로 상승
            self._notify_progress('안전 높이로 상승', 60)
            current_pos = self.robot.get_current_posx() if hasattr(self.robot, 'get_current_posx') else None
            
            if current_pos and len(current_pos) >= 6:
                safe_pos = list(current_pos)
                safe_pos[2] = max(safe_pos[2], HOME_POSITION[2])  # HOME Z 높이로
                self._safe_movel(safe_pos, vel=VELOCITY_MOVE, acc=ACCEL_MOVE)
                time.sleep(0.3)
            
            # 2. 안전 Drop 위치로 이동 (높이 유지)
            self._notify_progress('Drop 위치로 이동', 70)
            approach_pos = SAFE_DROP_POSITION.copy()
            approach_pos[2] = HOME_POSITION[2]  # 안전 높이
            self._safe_movel(approach_pos, vel=VELOCITY_MOVE, acc=ACCEL_MOVE)
            time.sleep(0.3)
            
            # 3. Drop 위치로 하강
            self._notify_progress('물체 내려놓기', 80)
            self._safe_movel(SAFE_DROP_POSITION, vel=VELOCITY_MOVE/2, acc=ACCEL_MOVE/2)
            time.sleep(0.3)
            
            # 4. 그리퍼 열기
            if hasattr(self.robot, 'grip_open'):
                self.robot.grip_open()
                time.sleep(0.5)
            
            # 5. 안전 높이로 복귀
            self._safe_movel(approach_pos, vel=VELOCITY_MOVE, acc=ACCEL_MOVE)
            time.sleep(0.3)
            
            # 6. 그리퍼 닫기
            if hasattr(self.robot, 'grip_close'):
                self.robot.grip_close()
                time.sleep(0.3)
            
            # 7. 홈으로 이동
            self._notify_progress('홈 위치로 이동', 90)
            success = self._safe_movel(HOME_POSITION, vel=VELOCITY_MOVE, acc=ACCEL_MOVE)
            
            if success:
                self.node.get_logger().info('[Recovery] ✅ 물체 반납 후 홈 도착')
            
            return success
            
        except Exception as e:
            self.node.get_logger().error(f'[Recovery] place_and_go_home 예외: {e}')
            return False
    
    # =========================================
    # 자동 복구
    # =========================================
    @property
    def is_recovering(self) -> bool:
        """복구 중인지 여부"""
        return self._is_recovering
    
    @property
    def was_collision_recovery(self) -> bool:
        """마지막 복구가 충돌로 인한 것인지 (사이클 카운트 스킵용)"""
        return self._recovery_caused_by_collision
    
    def clear_collision_flag(self):
        """충돌 복구 플래그 클리어"""
        self._recovery_caused_by_collision = False
    
    def save_work_state(self, state: dict):
        """작업 상태 저장 (복구 후 재개용)"""
        self._saved_work_state = state
    
    def get_saved_work_state(self) -> Optional[dict]:
        """저장된 작업 상태 반환"""
        return self._saved_work_state
    
    def clear_saved_work_state(self):
        """저장된 작업 상태 삭제"""
        self._saved_work_state = None
    
    def auto_recover(self, max_attempts: int = 3) -> bool:
        """
        자동 복구 시퀀스 실행
        
        복구 시나리오:
        1. 그립 상태 → 안전 위치에 물체 반납 → 홈 이동
        2. 비그립 상태 → 홈 직행
        
        Returns:
            복구 성공 여부
        """
        if self._is_recovering:
            self.node.get_logger().warn('[Recovery] 이미 복구 중')
            return False
        
        self._is_recovering = True
        self._recovery_caused_by_collision = True
        success = False
        was_gripping = False
        
        try:
            # 그립 상태 확인
            if self.robot and hasattr(self.robot, 'is_gripping'):
                was_gripping = self.robot.is_gripping()
                grip_status = "🔴 물체 잡음" if was_gripping else "⚪ 빈 손"
                self.node.get_logger().info(f'[Recovery] 그립 상태: {grip_status}')
                self._write_log('info', f'그립 상태: {grip_status}')
            
            # Z 높이 확인 (바닥 충돌 판단)
            current_z = self._get_current_z()
            needs_jog = current_z is not None and current_z < RECOVERY_Z_THRESHOLD
            
            z_str = f'{current_z:.1f}mm' if current_z else 'N/A'
            case_type = '바닥 충돌' if needs_jog else '외부 충돌'
            
            self.node.get_logger().info('=' * 60)
            self.node.get_logger().info(f'[Recovery] 자동 복구 시작 - {case_type}, Z={z_str}')
            self._write_log('info', f'🚨 자동 복구 시작 - {case_type}, Z={z_str}')
            if was_gripping:
                self.node.get_logger().info('[Recovery] → 물체 반납 후 홈으로 이동')
                self._write_log('info', '→ 물체 반납 후 홈으로 이동')
            else:
                self.node.get_logger().info('[Recovery] → 홈으로 직행')
                self._write_log('info', '→ 홈으로 직행')
            self.node.get_logger().info('=' * 60)
            
            for attempt in range(max_attempts):
                self.node.get_logger().info(f'[Recovery] 시도 {attempt + 1}/{max_attempts}')
                
                # 1. SAFE_STOP 리셋
                if not self.reset_safe_stop():
                    self.node.get_logger().warn('[Recovery] SAFE_STOP 리셋 실패 - 재시도')
                    time.sleep(0.5)
                    continue
                
                # 2. RECOVERY 진입
                if not self.enter_recovery():
                    self.node.get_logger().warn('[Recovery] RECOVERY 진입 실패 - 재시도')
                    time.sleep(0.5)
                    continue
                
                # 3. Jog Z+ (바닥 충돌 시)
                if needs_jog:
                    # 바닥 충돌 + 그립 상태면 먼저 그리퍼 열기 (물체 끼임 방지)
                    if was_gripping and self.robot and hasattr(self.robot, 'grip_open'):
                        self.node.get_logger().info('[Recovery] 바닥 충돌 - 그리퍼 열기')
                        self.robot.grip_open()
                        time.sleep(0.3)
                        was_gripping = False
                    
                    if not self.jog_up():
                        self.node.get_logger().warn('[Recovery] Jog 상승 실패 - 재시도')
                        time.sleep(0.5)
                        continue
                else:
                    self._notify_progress('Jog 생략', 50)
                
                # 4. RECOVERY 완료
                if not self.complete_recovery():
                    self.node.get_logger().warn('[Recovery] RECOVERY 완료 실패 - 재시도')
                    time.sleep(0.5)
                    continue
                
                # 5. RECOVERY 해제
                if not self.exit_recovery():
                    self.node.get_logger().warn('[Recovery] RECOVERY 해제 실패 - 재시도')
                    time.sleep(0.5)
                    continue
                
                # 6. 서보 ON
                if not self.servo_on():
                    self.node.get_logger().warn('[Recovery] 서보 ON 실패 - 재시도')
                    time.sleep(0.5)
                    continue
                
                # 7. 서비스 안정화 대기
                self.node.get_logger().info('[Recovery] 서비스 안정화 대기 (2초)...')
                time.sleep(2.0)
                
                # 🔴 최종 상태 확인 및 STANDBY 대기
                self.node.get_logger().info('[Recovery] 최종 상태 확인 중...')
                max_wait = 10.0  # 최대 10초 대기
                wait_start = time.time()
                final_state = None
                
                while (time.time() - wait_start) < max_wait:
                    final_state = self._get_robot_state()
                    if final_state is not None:
                        self.node.get_logger().info(f'[Recovery] 📊 현재 상태: {state_name(final_state)}')
                        
                        if final_state == RobotState.STANDBY:
                            self.node.get_logger().info('[Recovery] ✅ STANDBY 상태 확인 완료')
                            break
                        elif final_state == RobotState.SAFE_STOP:
                            self.node.get_logger().warn(f'[Recovery] ⏳ SAFE_STOP 상태 - STANDBY 대기 중... ({int(time.time() - wait_start)}초)')
                        elif final_state == RobotState.SAFE_OFF:
                            self.node.get_logger().error('[Recovery] ❌ SAFE_OFF 상태 - 복구 실패!')
                            recovery_success = False
                            break
                        else:
                            self.node.get_logger().warn(f'[Recovery] ⚠️ 예상치 못한 상태: {state_name(final_state)}')
                    
                    time.sleep(0.5)
                
                # 타임아웃 체크
                if final_state != RobotState.STANDBY:
                    self.node.get_logger().error(f'[Recovery] ❌ STANDBY 도달 실패! 최종 상태: {state_name(final_state) if final_state else "UNKNOWN"}')
                    self.node.get_logger().error('[Recovery] 🤖 로봇을 수동으로 STANDBY 상태로 만들어주세요!')
                    recovery_success = False
                
                # recovery_success가 False면 여기서 중단
                if not recovery_success:
                    self.node.get_logger().error('[Recovery] ❌ 복구 실패 - 재시도 중...')
                    time.sleep(1.0)
                    continue
                
                # ✅ 6단계 복구 시퀀스 완료
                self._notify_progress('복구 시퀀스 완료', 95)
                self.node.get_logger().info('✅ [Recovery] 6단계 복구 시퀀스 완료 (STANDBY 상태 달성)')
                recovery_success = True
                
                if recovery_success:
                    self._notify_progress('복구 완료', 100)
                    self.node.get_logger().info('✅ [Recovery] 복구 시퀀스 완료!')
                    
                    # 드라이버 재시작으로 상태 초기화 (movel 충돌 방지)
                    self.node.get_logger().warn('[Recovery] 🔄 복구 완료 후 드라이버 재시작...')
                    self._restart_driver()
                    
                    self.node.get_logger().info('[Recovery] ✅ 전체 복구 완료 (movel 제외)')
                    
                    success = True
                    break
                else:
                    self._notify_progress('복구 완료 (일부 실패)', 95)
                    self.node.get_logger().warn('⚠️ [Recovery] 복구 일부 실패')
                    success = True  # 상태 복구는 성공했으므로 true
                    break
            
            if not success:
                self.node.get_logger().error('[Recovery] 복구 실패')
                self._notify_progress('복구 실패', 0)
        
        except Exception as e:
            self.node.get_logger().error(f'[Recovery] 예외: {e}')
            success = False
        
        finally:
            self._is_recovering = False
            if self._on_complete:
                try:
                    self._on_complete(success, was_gripping)
                except Exception as e:
                    self.node.get_logger().error(f'완료 콜백 에러: {e}')
        
        return success
    
    def start_recovery_thread(self):
        """별도 스레드에서 복구 시작"""
        thread = threading.Thread(target=self.auto_recover, daemon=True)
        thread.start()
    
    def _restart_driver(self):
        """
        DSR Bringup 재시작 (복구 완료 후 상태 초기화용)
        
        프로세스:
        1. 기존 dsr_bringup 프로세스 강제 종료
        2. 2초 대기 (정리)
        3. dsr_bringup2 launch 파일 재실행 (백그라운드)
        4. 상태 강제 업데이트 (MOVING → 정상 상태)
        
        목적: 
        - MOVING 상태에서 멈춘 로봇 강제 리셋
        - 복구 플래그 리셋 준비 (정상 상태 감지 필요)
        """
        try:
            self.node.get_logger().warn('='*60)
            self.node.get_logger().warn('🔄 [Bringup] DSR Bringup 자동 재시작')
            self.node.get_logger().warn('='*60)
            
            self._notify_progress('Bringup 재시작 중...', 10)
            self._write_log('info', '[Bringup] 드라이버 재시작 시작')
            
            # 1. 기존 프로세스 강제 종료 (rviz 포함)
            self.node.get_logger().warn('[Bringup] 기존 프로세스 종료 중...')
            self._write_log('info', '기존 프로세스 종료 중 (dsr_bringup2, ros2_control, rviz2)')
            
            subprocess.run(['pkill', '-9', '-f', 'dsr_bringup2'], stderr=subprocess.DEVNULL)
            subprocess.run(['pkill', '-9', '-f', 'ros2_control_node'], stderr=subprocess.DEVNULL)
            subprocess.run(['pkill', '-9', '-f', 'robot_state_publisher'], stderr=subprocess.DEVNULL)
            subprocess.run(['pkill', '-9', '-f', 'rviz2'], stderr=subprocess.DEVNULL)
            subprocess.run(['killall', '-9', 'rviz2'], stderr=subprocess.DEVNULL)
            
            self.node.get_logger().info('[Bringup] 프로세스 종료 완료 - 2초 대기')
            time.sleep(2.0)
            
            # 2. FastDDS 공유 메모리 정리 (포트 충돌 방지)
            self.node.get_logger().info('[Bringup] FastDDS 공유 메모리 정리...')
            subprocess.run(['bash', '-c', 'rm -f /dev/shm/fastrtps_* 2>/dev/null'], stderr=subprocess.DEVNULL)
            time.sleep(0.5)
            
            # 3. dsr_bringup2 launch 재실행 (백그라운드)
            self.node.get_logger().warn('[Bringup] launch 백그라운드 재시작 중...')
            self._write_log('info', 'launch 재시작 명령 발송')
            
            cmd = [
                'bash', '-c',
                'source /home/rokey/ros2_ws/install/setup.bash && '
                'source /home/rokey/rokey/DUM-E/ros2_ws/install/setup.bash && '
                'ros2 launch dsr_bringup2 dsr_bringup2_rviz.launch.py '
                'mode:=real host:=192.168.1.100 port:=12345 model:=m0609 gui:=false '
                '> /tmp/dsr_bringup.log 2>&1 &'
            ]
            subprocess.Popen(cmd)
            
            self.node.get_logger().info('[Bringup] ✅ 재시작 명령 발송 완료')
            self._notify_progress('Bringup 재시작 명령 발송', 100)
            self._write_log('info', '✅ Bringup 재시작 완료')
            
            # 🔴 재시작 후 안정화 대기 (3초)
            self.node.get_logger().info('[Bringup] 시스템 안정화 대기 중... (3초)')
            time.sleep(3.0)
            
            # 🔴 재시작 후 상태 확인
            final_state = self._get_robot_state()
            if final_state is not None:
                self.node.get_logger().info(f'[Bringup] 📊 재시작 후 로봇 상태: {state_name(final_state)}')
                if final_state == RobotState.SAFE_STOP:
                    self.node.get_logger().error('[Bringup] ❌ 재시작 후에도 SAFE_STOP 상태!')
                elif final_state == RobotState.SAFE_OFF:
                    self.node.get_logger().warn('[Bringup] ⚠️ 재시작 후 SAFE_OFF 상태 - 서보 ON 필요')
                elif final_state == RobotState.STANDBY:
                    self.node.get_logger().info('[Bringup] ✅ STANDBY 상태 - 정상')
            else:
                self.node.get_logger().warn('[Bringup] ⚠️ 상태 확인 실패 (서비스 대기 중)')
            
            return True
        
        except Exception as e:
            self.node.get_logger().error(f'[Bringup] 재시작 실패: {e}')
            self._write_log('error', f'[Bringup] 재시작 실패: {e}')
            return False
