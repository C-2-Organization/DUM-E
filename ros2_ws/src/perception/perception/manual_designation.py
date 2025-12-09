import cv2
import time
import numpy as np
import rclpy
from rclpy.node import Node
from scipy.spatial.transform import Rotation

from utils.realsense import ImgNode
from utils.onrobot import RG
import DR_init
import os
from ament_index_python.packages import get_package_share_directory

# for single robot
ROBOT_ID = "dsr01"
ROBOT_MODEL = "m0609"
VELOCITY, ACC = 60, 60

DR_init.__dsr__id = ROBOT_ID
DR_init.__dsr__model = ROBOT_MODEL

GRIPPER_NAME = "rg2"
TOOLCHARGER_IP = "192.168.1.1"
TOOLCHARGER_PORT = "502"

# =============================
# 공용 속도 / 가속도 프로파일
# =============================

# 직선/포즈 이동용 (movel)
LIN_VEL = [150.0, 300.0]   # v1, v2
LIN_ACC = [150.0, 150.0]   # a1, a2

# 관절 이동용 (movej)
JNT_VEL = 100.0            # deg/s
JNT_ACC = 150.0            # deg/s^2

CUSTOM_HOME_JOINT = [0, 0, 90, 0, 90, 0]

class ManualDesignation(Node):
    def __init__(self):
        from DSR_ROBOT2 import posj
        super().__init__("manual_designation")
        self.intrinsics = None
        self.img_node = ImgNode()

        pkg_share = get_package_share_directory('perception')
        calib_path = os.path.join(pkg_share, 'config', 'T_gripper2camera.npy')
        self.gripper2cam = np.load(calib_path)

        self.JReady = posj([0, 0, 90, 0, 90, -90])
        self.gripper = RG(GRIPPER_NAME, TOOLCHARGER_IP, TOOLCHARGER_PORT)

    def ensure_intrinsics(self, timeout_sec: float = 2.0) -> bool:
        """intrinsics가 None이면 timeout 안에서 재시도해서 채움"""
        if self.intrinsics is not None:
            return True

        start = time.time()
        while time.time() - start < timeout_sec:
            intr = self.img_node.get_camera_intrinsic()
            if intr is not None:
                self.intrinsics = intr
                self.get_logger().info(f"Camera intrinsics loaded: {self.intrinsics}")
                return True
            time.sleep(0.05)

        self.get_logger().error("Failed to get camera intrinsics within timeout")
        return False

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if not self.ensure_intrinsics():
                self.get_logger().warn("intrinsics not ready, ignore click")
                return

            depth_frame = self.img_node.get_depth_frame()
            retry = 0
            while (depth_frame is None or np.all(depth_frame == 0)) and retry < 30:
                self.get_logger().info("retry get depth img")
                time.sleep(0.05)
                depth_frame = self.img_node.get_depth_frame()
                retry += 1

            if depth_frame is None or np.all(depth_frame == 0):
                self.get_logger().error("depth frame not ready, ignore click")
                return

            print(f"img cordinate: ({x}, {y})")
            z = self.get_depth_value(x, y, depth_frame)
            if z is None or z == 0:
                self.get_logger().warn("invalid depth value, ignore click")
                return

            camera_center_pos = self.get_camera_pos(x, y, z, self.intrinsics)
            print(f"camera cordinate: ({camera_center_pos})")

            robot_coordinate = self.transform_to_base(camera_center_pos)
            print(f"robot cordinate: ({robot_coordinate})")

            self.pick_and_drop(*robot_coordinate)
            print("=" * 100)

    def get_camera_pos(self, center_x, center_y, center_z, intrinsics):
        camera_x = (center_x - intrinsics["ppx"]) * center_z / intrinsics["fx"]
        camera_y = (center_y - intrinsics["ppy"]) * center_z / intrinsics["fy"]
        camera_z = center_z

        return (camera_x, camera_y, camera_z)

    def get_robot_pose_matrix(self, x, y, z, rx, ry, rz):
        R = Rotation.from_euler("ZYZ", [rx, ry, rz], degrees=True).as_matrix()
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = [x, y, z]
        return T

    def pick_and_drop(self, x, y, z):
        from DSR_ROBOT2 import (
            movel,
            movej,
        )
        current_pos = get_current_posx()[0]
        # TODO(Insun35): remove hard code Z offset
        pick_pos = posx([x, y, z+210, current_pos[3], current_pos[4], current_pos[5]])
        movel(
            pick_pos,
            vel=LIN_VEL,
            acc=LIN_ACC,
            mod=DR_MV_MOD_ABS,
            ra=DR_MV_RA_DUPLICATE,
        )
        self.gripper.close_gripper()
        wait(1)

        movej(
            CUSTOM_HOME_JOINT,
            vel=JNT_VEL,
            acc=JNT_ACC,
            mod=DR_MV_MOD_ABS,
            ra=DR_MV_RA_DUPLICATE,
        )
        self.gripper.open_gripper()
        wait(1)

    def transform_to_base(self, camera_coords):
        """
        Converts 3D coordinates from the camera coordinate system
        to the robot's base coordinate system.
        """
        coord = np.append(np.array(camera_coords), 1)

        base_pos = get_current_posx()[0]
        base2gripper = self.get_robot_pose_matrix(*base_pos)

        base2cam = base2gripper @ self.gripper2cam
        td_coord = np.dot(base2cam, coord)

        return td_coord[:3]

    def open_img_node(self):
        img = self.img_node.get_color_frame()

        if img is None or img.size == 0:
            self.get_logger().warn("color frame not ready yet")
            return

        cv2.setMouseCallback("Webcam", self.mouse_callback, img)
        cv2.imshow("Webcam", img)

    def get_depth_value(self, center_x, center_y, depth_frame):
        height, width = depth_frame.shape
        if 0 <= center_x < width and 0 <= center_y < height:
            depth_value = depth_frame[center_y, center_x]
            return depth_value
        self.get_logger().warn(f"out of image range: {center_x}, {center_y}")
        return None


def main(args=None):
    global get_current_posx, movej, movel, wait, DR_MV_MOD_ABS, DR_MV_RA_DUPLICATE, set_tool, set_tcp, posx, posj

    rclpy.init(args=args)

    # DSR node
    dsr_node = rclpy.create_node("dsr_example_demo_py", namespace=ROBOT_ID)
    DR_init.__dsr__node = dsr_node

    from DSR_ROBOT2 import (
        get_current_posx as _get_current_posx,
        movej as _movej,
        movel as _movel,
        wait as _wait,
        DR_MV_MOD_ABS as _DR_MV_MOD_ABS,
        DR_MV_RA_DUPLICATE as _DR_MV_RA_DUPLICATE,
        set_tool as _set_tool,
        set_tcp as _set_tcp,
    )
    from DR_common2 import posx as _posx, posj as _posj

    get_current_posx = _get_current_posx
    movej = _movej
    movel = _movel
    wait = _wait
    DR_MV_MOD_ABS = _DR_MV_MOD_ABS
    DR_MV_RA_DUPLICATE = _DR_MV_RA_DUPLICATE
    set_tool = _set_tool
    set_tcp = _set_tcp
    posx = _posx
    posj = _posj

    node = ManualDesignation()

    from rclpy.executors import MultiThreadedExecutor
    from threading import Thread

    executor = MultiThreadedExecutor()
    executor.add_node(dsr_node)
    executor.add_node(node)
    executor.add_node(node.img_node)

    spin_thread = Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    cv2.namedWindow("Webcam")

    try:
        movej(
            CUSTOM_HOME_JOINT,
            vel=JNT_VEL,
            acc=JNT_ACC,
            mod=DR_MV_MOD_ABS,
            ra=DR_MV_RA_DUPLICATE,
        )

        while rclpy.ok():
            node.open_img_node()

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
    finally:
        cv2.destroyAllWindows()
        executor.shutdown()
        node.destroy_node()
        node.img_node.destroy_node()
        dsr_node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
