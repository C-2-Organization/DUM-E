#!/usr/bin/env python3
"""
Robot State Care Node

Doosan 드라이버 상태를 모니터링하고 SAFE_STOP 시 자동 복구,
드라이버 무응답 시 자동 재시작을 수행합니다.
"""

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from dsr_msgs2.srv import GetRobotState
import time
import threading
import subprocess
import socket
import os
import sys

from dum_e_motion.recovery import CollisionRecovery, RobotState


ROBOT_ID = 'dsr01'


def log_print(msg: str, level='info'):
    """모든 출력을 로거에도 기록"""
    print(msg, flush=True)
    sys.stdout.flush()


def state_name(state) -> str:
    if state is None:
        return 'UNKNOWN'
    if isinstance(state, RobotState):
        return state.name
    try:
        return RobotState(state).name
    except Exception:
        return f'UNKNOWN({state})'


class RobotStateCareNode(Node):
    def __init__(self):
        super().__init__('robot_state_care_node')
        self.callback_group = ReentrantCallbackGroup()

        prefix = f'/{ROBOT_ID}'

        self.cli_state = self.create_client(
            GetRobotState, f'{prefix}/system/get_robot_state',
            callback_group=self.callback_group
        )

        self.recovery = CollisionRecovery(
            node=self,
            robot_controller=None,
            callback_group=self.callback_group
        )

        self.recovery.set_progress_callback(self._on_progress)
        self.recovery.set_complete_callback(self._on_complete)

        self._previous_state = None
        self._is_recovering = False
        # ROS 파라미터로 튜닝 가능
        self.declare_parameter('monitor_interval', 0.2)
        self.declare_parameter('failure_threshold', 8)
        self.declare_parameter('initial_safe_stop_grace', 1.5)

        self._monitor_interval = float(self.get_parameter('monitor_interval').value)
        self._consecutive_failures = 0
        self._failure_threshold = int(self.get_parameter('failure_threshold').value)
        self._seen_normal_state = False
        self._restarting_driver = False
        self._allow_initial_safe_stop_recovery = True
        self._initial_safe_stop_grace = float(self.get_parameter('initial_safe_stop_grace').value)
        self._safe_stop_start_ts = None

        self._print_banner()
        self._wait_for_services()

        self.create_timer(self._monitor_interval, self._monitor_callback)

    def _on_progress(self, message: str, percent: int):
        bar_length = 30
        filled = int(bar_length * percent / 100)
        bar = '█' * filled + '░' * (bar_length - filled)
        print(f'[{bar}] {percent:3d}% - {message}', end='\r', flush=True)

    def _on_complete(self, success: bool, was_gripping: bool):
        if success:
            print(f'\n✅ 복구 성공!', flush=True)
            self.get_logger().info('[Care] ✅ 복구 완료')
        else:
            print(f'\n❌ 복구 실패!', flush=True)
            self.get_logger().error('[Care] ❌ 복구 실패')

    def _print_banner(self):
        print('\n' + '=' * 60, flush=True)
        print('🛡️  Robot State Care Node', flush=True)
        print('=' * 60, flush=True)
        print('  • 로봇 상태 모니터링 + 자동 복구', flush=True)
        print('  • 드라이버 무응답 자동 재시작', flush=True)
        print('=' * 60 + '\n', flush=True)

    def _wait_for_services(self):
        print('⏳ DSR 드라이버 서비스 연결 대기 중...', flush=True)
        if not self.cli_state.wait_for_service(timeout_sec=10.0):
            print('❌ get_robot_state 서비스 연결 실패! (robot-real 실행 확인)', flush=True)
            raise SystemExit(1)
        print('✅ 서비스 연결 완료! 모니터링 시작...\n', flush=True)

    def _get_robot_state(self):
        if not self.cli_state.service_is_ready():
            return None
        req = GetRobotState.Request()
        future = self.cli_state.call_async(req)
        start = time.time()
        # 더 짧은 타임아웃과 더 빠른 폴링으로 응답성 향상
        while not future.done() and (time.time() - start) < 0.5:
            time.sleep(0.01)
        if future.done() and future.result():
            try:
                return RobotState(future.result().robot_state)
            except Exception:
                return None
        return None

    def _monitor_callback(self):
        if self._is_recovering:
            return

        try:
            state = self._get_robot_state()
        except Exception as e:
            self.get_logger().error(f'[Care] 상태 조회 예외: {e}')
            self._consecutive_failures += 1
            return

        if state is None:
            self._consecutive_failures += 1
            if self._consecutive_failures >= self._failure_threshold and not self._restarting_driver:
                if self._consecutive_failures == self._failure_threshold:
                    print(f'\n💀 DSR 드라이버 응답 없음 ({self._failure_threshold}회 연속 실패)! 자동 재시작...', flush=True)
                    self.get_logger().error(f'[Care] 드라이버 응답 없음 ({self._consecutive_failures}회) - 자동 재시작')
                    self._restart_driver()
            return
        else:
            self._consecutive_failures = 0
            self._restarting_driver = False

        if state != RobotState.SAFE_STOP:
            self._seen_normal_state = True
            self._safe_stop_start_ts = None  # SAFE_STOP 벗어나면 타이머 초기화

        if state != self._previous_state:
            print(f'📊 상태 변경: {state_name(self._previous_state)} → {state_name(state)}', flush=True)
            self.get_logger().info(f'[Care] 상태: {state_name(state)}')
            self._previous_state = state

        # SAFE_STOP 처리 로직
        if state == RobotState.SAFE_STOP:
            # 1) 정상 상태를 한 번이라도 본 뒤의 SAFE_STOP → 즉시 복구
            if self._seen_normal_state:
                print(f'\n⚠️  충돌 감지! SAFE_STOP → 복구 시작', flush=True)
                self.get_logger().warn('[Care] SAFE_STOP 감지 - 복구 시작')
                self._seen_normal_state = False  # 복구 시작 시 플래그 리셋 (다음 충돌도 감지 가능)
                self._execute_recovery()
                return

            # 2) 부팅 직후부터 SAFE_STOP 상태가 지속되는 경우 → 유예 시간 경과 시 복구 허용(옵션)
            if self._allow_initial_safe_stop_recovery:
                now = time.time()
                if self._safe_stop_start_ts is None:
                    self._safe_stop_start_ts = now
                    print('⏳ 부팅 직후 SAFE_STOP 감지 - 유예 카운트다운 시작', flush=True)
                elif (now - self._safe_stop_start_ts) >= self._initial_safe_stop_grace:
                    print(f'\n⚠️  초기 SAFE_STOP 지속({self._initial_safe_stop_grace:.0f}s 경과) → 복구 시작', flush=True)
                    self.get_logger().warn('[Care] 초기 SAFE_STOP 지속 - 복구 시작')
                    self._execute_recovery()
                    # 다음 이벤트를 위해 타이머/플래그 리셋
                    self._safe_stop_start_ts = None
                    self._seen_normal_state = False
                    return

    def _restart_driver(self):
        self._restarting_driver = True

        def _restart_thread():
            try:
                my_pid = os.getpid()  # 현재 프로세스 ID 저장
                print(f'🔄 Step 1/5: 기존 프로세스 종료... (robot_state_care PID={my_pid} 제외)', flush=True)
                
                # RViz 먼저 강제 종료 (여러 번 시도)
                print('   📺 RViz 종료 중...', flush=True)
                for _ in range(3):  # 3번 반복으로 확실히 종료
                    subprocess.run(['pkill', '-9', 'rviz2'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    time.sleep(0.3)
                
                    # Launch 프로세스 종료 (dum_e_bringup 전체를 타겟)
                    subprocess.run(['pkill', '-9', '-f', 'ros2 launch dum_e_bringup dum_e_bringup.launch.py'],
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    # 혹시 남아 있을 수 있는 dsr_bringup2 단독 실행도 종료
                    subprocess.run(['pkill', '-9', '-f', 'dsr_bringup2_rviz.launch.py'],
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    subprocess.run(['pkill', '-9', '-f', 'ros2 launch dsr_bringup2'],
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                time.sleep(0.5)
                
                # RViz 명시적 종료
                print('   🖼️  RViz 종료 중...', flush=True)
                subprocess.run(['pkill', '-9', 'rviz2'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                time.sleep(0.5)
                
                # Doosan 드라이버 관련 모든 프로세스 종료 (여러 번 실행)
                print('   🤖 로봇 드라이버 종료 중...', flush=True)
                for _ in range(2):  # 2번 반복해서 확실히 종료
                    subprocess.run(['pkill', '-9', '-f', 'dsr_bringup'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    subprocess.run(['pkill', '-9', '-f', 'ros2_control'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    subprocess.run(['pkill', '-9', '-f', 'controller_manager'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    subprocess.run(['pkill', '-9', 'robot_state_publisher'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    subprocess.run(['pkill', '-9', 'ros2_control_node'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    subprocess.run(['pkill', '-9', '-f', 'dsr_hw_interface'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    subprocess.run(['pkill', '-9', '-f', 'dsr_controller'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    subprocess.run(['pkill', '-9', '-f', 'joint_state_broadcaster'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    time.sleep(0.5)
                
                # /dsr01 네임스페이스의 모든 노드 강제 종료 (robot_state_care 제외)
                print('   🧹 /dsr01 네임스페이스 정리 중...', flush=True)
                subprocess.run(['bash', '-c', f'ps aux | grep "/dsr01" | grep -v grep | grep -v robot_state_care | awk \'{{print $2}}\' | xargs -r kill -9'], 
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                time.sleep(2)

                print('🔄 Step 2/5: FastDDS cleanup...', flush=True)
                subprocess.run(['rm', '-rf', '/tmp/fastrtps*'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                subprocess.run(['rm', '-rf', '/tmp/launch_params_*'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                time.sleep(1)

                print('🔄 Step 3/5: 로봇 컨트롤러 대기 (192.168.1.100:12345)...', flush=True)
                robot_ready = False
                for i in range(60):
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(1)
                    result = sock.connect_ex(('192.168.1.100', 12345))
                    sock.close()
                    if result == 0:
                        robot_ready = True
                        print(f'✅ 로봇 컨트롤러 준비 완료 ({i+1}초)', flush=True)
                        break
                    time.sleep(1)
                if not robot_ready:
                    print('❌ 로봇 컨트롤러 연결 실패 (타임아웃)', flush=True)
                    self._restarting_driver = False
                    return

                time.sleep(3)

                print('🔄 Step 4/5: 통합 런치 재시작 (dum_e_bringup.launch.py)...', flush=True)
                # 백그라운드에서 bringup 실행 (로그 파일 저장)
                log_file = '/tmp/dsr_bringup_restart.log'
                with open(log_file, 'w') as f:
                    f.write(f'Restart initiated at {time.strftime("%Y-%m-%d %H:%M:%S")}\n')
                
                restart_cmd = (
                    'source /opt/ros/humble/setup.bash && '
                    'source /home/rokey/DUM-E/ros2_ws/install/setup.bash && '
                    'ros2 launch dum_e_bringup dum_e_bringup.launch.py '
                    'doosan_mode:=real doosan_host:=192.168.1.100 doosan_port:=12345 doosan_model:=m0609'
                )
                
                # nohup으로 백그라운드 실행 (로그 파일에 저장)
                subprocess.Popen(
                    f'nohup bash -c "{restart_cmd}" >> {log_file} 2>&1 &',
                    shell=True,
                    preexec_fn=os.setpgrp
                )

                print('🔄 Step 5/5: 서비스 대기 (최대 30초)...', flush=True)
                for i in range(30):
                    if self.cli_state.wait_for_service(timeout_sec=1.0):
                        print(f'✅ robot-real 재시작 완료! ({i+1}초)', flush=True)
                        self._restarting_driver = False
                        self._consecutive_failures = 0
                        self._seen_normal_state = False
                        return
                print('❌ 서비스 복구 실패 (타임아웃)', flush=True)
            except Exception as e:
                print(f'❌ 재시작 실패: {e}', flush=True)
                import traceback
                traceback.print_exc(file=sys.stdout)
                sys.stdout.flush()
            finally:
                self._restarting_driver = False

        threading.Thread(target=_restart_thread, daemon=False).start()

    def _execute_recovery(self):
        self._is_recovering = True
        print('\n' + '=' * 60, flush=True)
        print('🔧 자동 복구 시작 (Recovery.auto_recover())', flush=True)
        print('=' * 60, flush=True)

        def _recovery_thread():
            try:
                print('⏳ auto_recover() 실행 중...', flush=True)
                success = self.recovery.auto_recover(max_attempts=3)
                if success:
                    self.get_logger().info('[Care] ✅ 복구 완료')
                    print('\n✅ 복구 완료! 모니터링 재개...\n', flush=True)
                else:
                    self.get_logger().error('[Care] ❌ 복구 실패')
                    print('\n❌ 복구 실패! 모니터링 재개...\n', flush=True)
            except Exception as e:
                self.get_logger().error(f'[Care] 복구 예외: {e}')
                print(f'\n⚠️  복구 중 예외 발생: {e}', flush=True)
                import traceback
                traceback.print_exc(file=sys.stdout)
                sys.stdout.flush()
                print('모니터링 재개...\n', flush=True)
            finally:
                self._is_recovering = False  # 복구 종료, 모니터링 재개

        recovery_thread = threading.Thread(target=_recovery_thread, daemon=False)
        recovery_thread.start()


def main(args=None):
    rclpy.init(args=args)
    node = RobotStateCareNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    
    print('\n🛡️  Robot State Care Node가 백그라운드에서 계속 실행됩니다...')
    print('   Ctrl+C로 종료할 수 있습니다.\n')
    
    try:
        executor.spin()  # 무한 루프 - 계속 모니터링
    except KeyboardInterrupt:
        print('\n\n👋 사용자가 종료를 요청했습니다.')
    except Exception as e:
        print(f'\n\n⚠️  예외 발생: {e}')
        print('노드를 재시작해주세요.')
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
