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

from dum_e_motion.recovery import CollisionRecovery, RobotState


ROBOT_ID = 'dsr01'


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
        self._monitor_interval = 0.5
        self._consecutive_failures = 0
        self._seen_normal_state = False
        self._restarting_driver = False

        self._print_banner()
        self._wait_for_services()

        self.create_timer(self._monitor_interval, self._monitor_callback)

    def _on_progress(self, message: str, percent: int):
        bar_length = 30
        filled = int(bar_length * percent / 100)
        bar = '█' * filled + '░' * (bar_length - filled)
        print(f'[{bar}] {percent:3d}% - {message}', end='\r')

    def _on_complete(self, success: bool, was_gripping: bool):
        if success:
            print(f'\n✅ 복구 성공!')
            self.get_logger().info('[Care] ✅ 복구 완료')
        else:
            print(f'\n❌ 복구 실패!')
            self.get_logger().error('[Care] ❌ 복구 실패')

    def _print_banner(self):
        print('\n' + '=' * 60)
        print('🛡️  Robot State Care Node')
        print('=' * 60)
        print('  • 로봇 상태 모니터링 + 자동 복구')
        print('  • 드라이버 무응답 자동 재시작')
        print('=' * 60 + '\n')

    def _wait_for_services(self):
        print('⏳ DSR 드라이버 서비스 연결 대기 중...')
        if not self.cli_state.wait_for_service(timeout_sec=10.0):
            print('❌ get_robot_state 서비스 연결 실패! (robot-real 실행 확인)')
            raise SystemExit(1)
        print('✅ 서비스 연결 완료! 모니터링 시작...\n')

    def _get_robot_state(self):
        if not self.cli_state.service_is_ready():
            return None
        req = GetRobotState.Request()
        future = self.cli_state.call_async(req)
        start = time.time()
        while not future.done() and (time.time() - start) < 1.0:
            time.sleep(0.02)
        if future.done() and future.result():
            try:
                return RobotState(future.result().robot_state)
            except Exception:
                return None
        return None

    def _monitor_callback(self):
        if self._is_recovering:
            return

        state = self._get_robot_state()

        if state is None:
            self._consecutive_failures += 1
            if self._consecutive_failures >= 5 and not self._restarting_driver:
                if self._consecutive_failures == 5:
                    print('\n💀 DSR 드라이버 응답 없음! 자동 재시작...')
                    self.get_logger().error('[Care] 드라이버 응답 없음 - 자동 재시작')
                    self._restart_driver()
            return
        else:
            self._consecutive_failures = 0
            self._restarting_driver = False

        if state != RobotState.SAFE_STOP:
            self._seen_normal_state = True

        if state != self._previous_state:
            print(f'📊 상태 변경: {state_name(self._previous_state)} → {state_name(state)}')
            self.get_logger().info(f'[Care] 상태: {state_name(state)}')
            self._previous_state = state

        if state == RobotState.SAFE_STOP and self._seen_normal_state:
            print(f'\n⚠️  충돌 감지! SAFE_STOP → 복구 시작')
            self.get_logger().warn('[Care] SAFE_STOP 감지 - 복구 시작')
            self._execute_recovery()

    def _restart_driver(self):
        self._restarting_driver = True

        def _restart_thread():
            try:
                print('🔄 Step 1/5: 기존 프로세스 종료...')
                subprocess.run(['pkill', '-9', '-f', 'dsr_bringup'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                subprocess.run(['pkill', '-9', '-f', 'ros2_control'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                time.sleep(2)

                print('🔄 Step 2/5: FastDDS cleanup...')
                subprocess.run(['rm', '-rf', '/tmp/fastrtps*'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                time.sleep(1)

                print('🔄 Step 3/5: 로봇 컨트롤러 대기 (192.168.1.100:12345)...')
                robot_ready = False
                for i in range(60):
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(1)
                    result = sock.connect_ex(('192.168.1.100', 12345))
                    sock.close()
                    if result == 0:
                        robot_ready = True
                        print(f'✅ 로봇 컨트롤러 준비 완료 ({i+1}초)')
                        break
                    time.sleep(1)
                if not robot_ready:
                    print('❌ 로봇 컨트롤러 연결 실패 (타임아웃)')
                    self._restarting_driver = False
                    return

                time.sleep(3)

                print('🔄 Step 4/5: robot-real 재시작...')
                subprocess.Popen(
                    ['bash', '-c', 'source /home/rokey/ros2_ws/install/setup.bash && '
                                   'source /home/rokey/rokey/DUM-E/ros2_ws/install/setup.bash && '
                                   'ros2 launch dsr_bringup2 dsr_bringup2_rviz.launch.py '
                                   'mode:=real host:=192.168.1.100 port:=12345 model:=m0609 gui:=false'],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    preexec_fn=os.setpgrp
                )

                print('🔄 Step 5/5: 서비스 대기 (최대 30초)...')
                for i in range(30):
                    if self.cli_state.wait_for_service(timeout_sec=1.0):
                        print(f'✅ robot-real 재시작 완료! ({i+1}초)')
                        self._restarting_driver = False
                        self._consecutive_failures = 0
                        self._seen_normal_state = False
                        return
                print('❌ 서비스 복구 실패 (타임아웃)')
            except Exception as e:
                print(f'❌ 재시작 실패: {e}')
            finally:
                self._restarting_driver = False

        threading.Thread(target=_restart_thread, daemon=False).start()

    def _execute_recovery(self):
        self._is_recovering = True
        print('\n' + '=' * 60)
        print('🔧 자동 복구 시작 (Recovery.auto_recover())')
        print('=' * 60)

        def _recovery_thread():
            try:
                success = self.recovery.auto_recover(max_attempts=3)
                if success:
                    self.get_logger().info('[Care] ✅ 복구 완료')
                else:
                    self.get_logger().error('[Care] ❌ 복구 실패')
            except Exception as e:
                self.get_logger().error(f'[Care] 복구 예외: {e}')
            finally:
                self._is_recovering = False

        threading.Thread(target=_recovery_thread, daemon=False).start()


def main(args=None):
    rclpy.init(args=args)
    node = RobotStateCareNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        print('\n\n👋 Robot State Care Node 종료')
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
