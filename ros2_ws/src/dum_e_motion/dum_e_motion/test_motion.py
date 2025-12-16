#!/usr/bin/env python3
"""
X방향 왕복 운동 테스트
X축으로 +50mm, -50mm 왕복 운동을 수행합니다.
"""
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from dsr_msgs2.srv import MoveJoint, MoveLine, GetCurrentPosx, GetRobotState
import time
class TestXMotion(Node):
    def __init__(self):
        super().__init__('test_x_motion')
        # Callback group for service calls
        self.callback_group = ReentrantCallbackGroup()
        # Service clients
        self.move_client = self.create_client(
            MoveLine,
            '/dsr01/motion/move_line',
            callback_group=self.callback_group
        )
        self.get_pos_client = self.create_client(
            GetCurrentPosx,
            '/dsr01/aux_control/get_current_posx',
            callback_group=self.callback_group
        )
        self.get_logger().info(':wrench: X방향 왕복 운동 테스트 노드 시작')
        # Wait for services
        self.get_logger().info('서비스 대기 중...')
        if not self.move_client.wait_for_service(timeout_sec=10.0):
            self.get_logger().error(':x: move_line 서비스를 찾을 수 없습니다')
            raise RuntimeError('move_line service not available')
        if not self.get_pos_client.wait_for_service(timeout_sec=10.0):
            self.get_logger().error(':x: get_current_posx 서비스를 찾을 수 없습니다')
            raise RuntimeError('get_current_posx service not available')
        self.get_logger().info(':white_check_mark: 서비스 연결 완료')
    def get_current_position(self):
        """현재 위치 가져오기"""
        req = GetCurrentPosx.Request()
        req.ref = 0  # Base reference
        self.get_logger().info('GetCurrentPosx 서비스 호출 중...')
        future = self.get_pos_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)
        if future.result() is not None:
            result = future.result()
            self.get_logger().info(f'서비스 응답: success={result.success}')
            self.get_logger().info(f'task_pos_info 길이: {len(result.task_pos_info)}')
            # task_pos_info는 Float64MultiArray의 리스트
            if result.success and len(result.task_pos_info) > 0:
                pos_data = list(result.task_pos_info[0].data[:6])
                self.get_logger().info(f'위치 데이터: {pos_data}')
                return pos_data
            else:
                self.get_logger().error(f'위치 가져오기 실패: success={result.success}')
        else:
            self.get_logger().error('서비스 응답이 None입니다 (타임아웃 또는 실패)')
        return None
    def move_line(self, positions, vel=120.0, acc=160.0):
        """직선 이동 (태스크 좌표계)"""
        req = MoveLine.Request()
        req.pos = positions
        req.vel = [vel, vel]  # task velocity
        req.acc = [acc, acc]  # task acceleration
        req.time = 0.0
        req.radius = 0.0
        req.ref = 0  # Base reference
        req.mode = 0  # Absolute mode
        req.blend_type = 0
        req.sync_type = 0
        future = self.move_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=30.0)
        if future.result() is not None:
            return future.result().success
        return False
    def run_test(self):
        """왕복 운동 실행"""
        self.get_logger().info('현재 위치 확인 중...')
        # Get current position
        current_pos = self.get_current_position()
        if current_pos is None:
            self.get_logger().warn(':warning: GetCurrentPosx 실패 - 기본 위치 사용')
            # 기본 홈 위치 사용 (X, Y, Z, Rx, Ry, Rz)
            current_pos = [0.0, 200.0, 300.0, 0.0, 180.0, 0.0]
        self.get_logger().info(f'시작 위치: {[f"{p:.2f}" for p in current_pos]}')
        # Calculate target positions
        pos_plus_50 = current_pos.copy()
        pos_plus_50[0] += 50.0  # X + 50mm
        pos_minus_50 = current_pos.copy()
        pos_minus_50[0] -= 50.0  # X - 50mm
        self.get_logger().info('=' * 60)
        self.get_logger().info(':arrows_counterclockwise: 왕복 운동 시작')
        self.get_logger().info('=' * 60)
        try:
            cycle = 1
            while rclpy.ok():
                # Move to +50
                self.get_logger().info(f'[Cycle {cycle}] X + 50mm 이동 중...')
                if not self.move_line(pos_plus_50):
                    self.get_logger().error('이동 실패 (X+50)')
                    break
                self.get_logger().info(':white_check_mark: X + 50mm 도착')
                time.sleep(1.0)
                # Move to -50
                self.get_logger().info(f'[Cycle {cycle}] X - 50mm 이동 중...')
                if not self.move_line(pos_minus_50):
                    self.get_logger().error('이동 실패 (X-50)')
                    break
                self.get_logger().info(':white_check_mark: X - 50mm 도착')
                time.sleep(1.0)
                cycle += 1
        except KeyboardInterrupt:
            self.get_logger().info('사용자 중단')
        # Return to original position
        self.get_logger().info('원래 위치로 복귀 중...')
        self.move_line(current_pos)
        self.get_logger().info(':white_check_mark: 테스트 완료')
def main(args=None):
    rclpy.init(args=args)
    node = TestXMotion()
    try:
        node.run_test()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
if __name__ == '__main__':
    main()
