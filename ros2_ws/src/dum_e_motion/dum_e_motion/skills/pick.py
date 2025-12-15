# dum_e_motion/skills/pick.py
import json
import time
import math
import os
from typing import Tuple, Optional

import rclpy
from rclpy.node import Node
from rclpy.task import Future
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2

from dum_e_interfaces.msg import SkillCommand
from dum_e_motion.motion_context import MotionContext


# ==============================================================================
# 설정 및 상수
# ==============================================================================
PICK_CONF_TH = 0.3         # 신뢰도 임계값
GRIPPER_OFFSET = 0.238178  # 그리퍼 길이 [m]
MODEL_PATH = './src/dum_e_perception/models/ggcnn_weights.pt'

# 카메라 내부 파라미터, ``ros2 topic echo /camera/camera/depth/camera_info``의 k로 확인
CAM_K = {
    'fx': 387.5936584472656,
    'fy': 387.5936584472656,
    'cx': 322.38787841796875,
    'cy': 234.24562072753906
}

# ==============================================================================
# 1. GG-CNN 모델 정의
# ==============================================================================
class GGCNN(nn.Module):
    def __init__(self, input_channels=1):
        super(GGCNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=9, stride=3, padding=3)
        self.conv2 = nn.Conv2d(32, 16, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(16, 8, kernel_size=3, stride=2, padding=1)
        self.convt1 = nn.ConvTranspose2d(8, 8, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.convt2 = nn.ConvTranspose2d(8, 16, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.convt3 = nn.ConvTranspose2d(16, 32, kernel_size=9, stride=3, padding=3, output_padding=1)
        self.pos_output = nn.Conv2d(32, 1, kernel_size=2)
        self.cos_output = nn.Conv2d(32, 1, kernel_size=2)
        self.sin_output = nn.Conv2d(32, 1, kernel_size=2)
        self.width_output = nn.Conv2d(32, 1, kernel_size=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.convt1(x))
        x = F.relu(self.convt2(x))
        x = F.relu(self.convt3(x))
        pos = self.pos_output(x)
        cos = self.cos_output(x)
        sin = self.sin_output(x)
        width = self.width_output(x)
        return pos, cos, sin, width

# ==============================================================================
# 2. AI 모델 관리 (싱글톤 패턴)
# ==============================================================================
_ggcnn_model = None
_cv_bridge = CvBridge()
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_ggcnn_model(logger):
    """모델을 최초 1회만 로딩"""
    global _ggcnn_model
    if _ggcnn_model is not None:
        return _ggcnn_model

    logger.info(f"[GGCNN] Loading model from {MODEL_PATH} on {_device}...")
    model = GGCNN().to(_device)
    try:
        # map_location을 사용하여 CPU/GPU 호환성 확보
        weights = torch.load(MODEL_PATH, map_location=_device)
        model.load_state_dict(weights)
        model.eval()
        _ggcnn_model = model
        logger.info("[GGCNN] Model loaded successfully.")
    except Exception as e:
        logger.error(f"[GGCNN] Failed to load model: {e}")
        return None
    return _ggcnn_model

# ==============================================================================
# 3. 이미지 캡처 및 추론 헬퍼
# ==============================================================================
class ImageCapture:
    """토픽에서 이미지 1장을 기다려서 받아오는 클래스"""
    # 기존 raw depth 대신 aligned depth 사용 (좌표 정밀도 향상)
    def __init__(self, node: Node, topic_name='/camera/camera/aligned_depth_to_color/image_raw'):
        self.node = node
        self.future = Future()
        self.sub = node.create_subscription(Image, topic_name, self.callback, 1)
        
    def callback(self, msg):
        if not self.future.done():
            self.future.set_result(msg)

    def wait_for_image(self, timeout_sec=3.0) -> Optional[Image]:
        start_time = time.time()
        while time.time() - start_time < timeout_sec:
            if self.future.done():
                return self.future.result()
            time.sleep(0.01)
        return None
    
    def destroy(self):
        self.node.destroy_subscription(self.sub)

def predict_grasp(node: Node, model: GGCNN) -> Tuple[Optional[PoseStamped], Optional[float], Optional[float]]:
    """
    이미지 캡처 -> 전처리 -> 모델 추론 -> 좌표 계산
    Returns: (PoseStamped, Angle_Rad, Width_mm)
    """
    # 1. 이미지 캡처
    capturer = ImageCapture(node)
    msg = capturer.wait_for_image(timeout_sec=3.0)
    capturer.destroy()

    if msg is None:
        node.get_logger().error("[GGCNN] Failed to capture depth image (Timeout)")
        return None, None, None

    try:
        # 2. 전처리
        depth_img_raw = _cv_bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        depth_img_raw = np.nan_to_num(depth_img_raw)
        depth_img = cv2.medianBlur(depth_img_raw.astype(np.float32), 5)
        
        h, w = depth_img.shape
        
        # 마스킹 (이미지상의 불필요 영역 제거)
        down_masking = 100  # [조정] 이미지 하단 마스킹 높이
        left_masking = 10  # [조정] 이미지 좌측 마스킹 폭

        depth_img[h - down_masking : h, :] = 0
        depth_img[:, 0:left_masking] = 0
        
        # 그리퍼 z축 기준 거리 필터
        min_dist = 50.0     # [조정] 최소 거리
        max_dist = 1000.0  # [조정] 최대 거리

        mask_depth = (depth_img > min_dist) & (depth_img < max_dist)
        depth_img = np.where(mask_depth, depth_img, 0)

        # 모델 입력용 크롭 및 정규화
        img_crop = cv2.resize(depth_img, (300, 300))
        valid_pixels = img_crop[img_crop > 0]
        if len(valid_pixels) > 0:
            min_val, max_val = valid_pixels.min(), valid_pixels.max()
            img_crop = np.where(img_crop > 0, (img_crop - min_val) / (max_val - min_val + 1e-6), 0)
        else:
            img_crop = np.zeros_like(img_crop)

        # 3. 추론
        depth_tensor = torch.from_numpy(img_crop).float().unsqueeze(0).unsqueeze(0).to(_device)
        with torch.no_grad():
            pos_out, cos_out, sin_out, width_out = model(depth_tensor)

        pos_map = pos_out.cpu().squeeze().numpy()
        cos_map = cos_out.cpu().squeeze().numpy()
        sin_map = sin_out.cpu().squeeze().numpy()
        width_map = width_out.cpu().squeeze().numpy()

        # 4. 후처리 (최고점 찾기)
        pos_map = cv2.GaussianBlur(pos_map, (7, 7), 0)
        max_val = np.max(pos_map)

        if max_val < 0.2: # 신뢰도 너무 낮으면 실패 처리
            node.get_logger().warn(f"[GGCNN] Low confidence: {max_val:.2f}")
            return None, None, None

        best_point = np.unravel_index(np.argmax(pos_map), pos_map.shape) # (y, x)

        best_cos = cos_map[best_point]
        best_sin = sin_map[best_point]
        best_width = width_map[best_point]

        # 각도 계산
        angle_rad = np.arctan2(best_sin, best_cos) / 2.0
        
        # 좌표 복원 (300x300 -> 원본 해상도)
        scale_y = h / 300.0
        scale_x = w / 300.0
        curr_y = int(best_point[0] * scale_y)
        curr_x = int(best_point[1] * scale_x)
        
        # 깊이(Z) 가져오기
        curr_z = depth_img[curr_y, curr_x]
        if curr_z == 0:
            # 구멍난 픽셀이면 주변 평균 사용
            curr_z = np.mean(depth_img[max(0, curr_y-2):curr_y+3, max(0, curr_x-2):curr_x+3])

        # 5. 3D 좌표 변환 (Pixel -> Camera Coordinate)
        Z_c = float(curr_z) # mm
        X_c = (curr_x - CAM_K['cx']) * Z_c / CAM_K['fx']
        Y_c = (curr_y - CAM_K['cy']) * Z_c / CAM_K['fy']

        # 실제 너비 계산
        pixel_width = best_width * scale_x
        real_width_mm = pixel_width * Z_c / CAM_K['fx']

        # 6. PoseStamped 생성
        pose = PoseStamped()
        pose.header.frame_id = "camera_depth_optical_frame" 
        pose.header.stamp = node.get_clock().now().to_msg()
        
        # mm -> meter 변환
        pose.pose.position.x = X_c / 1000.0
        pose.pose.position.y = Y_c / 1000.0
        pose.pose.position.z = Z_c / 1000.0
        
        # Orientation (Z-rotation only)
        pose.pose.orientation.z = math.sin(angle_rad / 2.0)
        pose.pose.orientation.w = math.cos(angle_rad / 2.0)

        return pose, angle_rad, real_width_mm

    except Exception as e:
        node.get_logger().error(f"[GGCNN] Inference error: {e}")
        return None, None, None

# ==============================================================================
# 4. 로봇 모션 실행 함수
# ==============================================================================
def execute_pick_motion(ctx: MotionContext, x, y, z, target_yaw_rad=None):
    from DSR_ROBOT2 import DR_MV_MOD_ABS, DR_MV_RA_DUPLICATE, get_current_posx
    from DR_common2 import posx

    ctx.node.get_logger().info(f"[MOVE] Pick → base({x:.3f}, {y:.3f}, {z:.3f})")

    current_pos = get_current_posx()[0]
    
    # 목표 회전값 (Rz)
    next_rz = current_pos[5]
    if target_yaw_rad is not None:
        next_rz = math.degrees(target_yaw_rad)

    # 1. 그리퍼 오픈
    ctx.motion.open_gripper()
    ctx.motion.wait(0.5)

    approach_pos = posx([x, y, z, current_pos[3], current_pos[4], next_rz])

    # 2. 접근
    ctx.motion.movel(approach_pos, vel=ctx.LIN_VEL, acc=ctx.LIN_ACC, mod=DR_MV_MOD_ABS, ra=DR_MV_RA_DUPLICATE)

    # 3. 집기
    ctx.motion.close_gripper()
    ctx.motion.wait(0.8)

    # 4. 들어 올리기 (안전)
    lift_pos = approach_pos
    # [조정] 픽킹 후 상승 높이 (100mm)
    lift_pos[2] += 100 
    ctx.motion.movel(lift_pos, vel=ctx.LIN_VEL, acc=ctx.LIN_ACC, mod=DR_MV_MOD_ABS, ra=DR_MV_RA_DUPLICATE)

    # 5. 홈 이동
    ctx.motion.movej(ctx.CUSTOM_HOME_JOINT, vel=ctx.JNT_VEL, acc=ctx.JNT_ACC, mod=DR_MV_MOD_ABS, ra=DR_MV_RA_DUPLICATE)

# ==============================================================================
# 5. 메인 스킬 함수 (Entry Point)
# ==============================================================================
def run_pick_skill(cmd: SkillCommand, ctx: MotionContext) -> Tuple[bool, str, float, PoseStamped]:
    object_name = cmd.object_name.strip()
    
    cam_pose = None
    confidence = 0.0
    grasp_angle_rad = 0.0

    if not object_name:
        return False, "object_name is empty", 0.0, PoseStamped()

    ctx.node.get_logger().info(f"[PICK] Start Skill: '{object_name}'")

    # ------------------------------------------------------------------
    # CASE A: GG-CNN 모드 ('ggcnn' 또는 'auto')
    # ------------------------------------------------------------------
    if object_name.lower() in ['ggcnn', 'auto', 'any']:
        # 1. 모델 로드 (최초 1회만 수행됨)
        model = load_ggcnn_model(ctx.node.get_logger())
        if model is None:
            return False, "Failed to load GGCNN model", 0.0, PoseStamped()

        # 2. 추론 실행
        ctx.node.get_logger().info("[PICK] Capturing image & Running Inference...")
        pose, angle, width = predict_grasp(ctx.node, model)

        if pose is None:
            return False, "GGCNN detection failed (No object or Timeout)", 0.0, PoseStamped()
        
        cam_pose = pose
        confidence = 1.0
        grasp_angle_rad = angle
        ctx.node.get_logger().info(f"[PICK] GGCNN Result: W={width:.1f}mm, Ang={math.degrees(angle):.1f}deg")

    # ------------------------------------------------------------------
    # CASE B: 외부 Target Pose 사용
    # ------------------------------------------------------------------
    elif cmd.target_pose.header.frame_id:
        cam_pose = cmd.target_pose
        confidence = 1.0
        ctx.node.get_logger().info("[PICK] Use external target pose")

    # ------------------------------------------------------------------
    # CASE C: 기존 Perception (YOLO 등)
    # ------------------------------------------------------------------
    else:
        pose_resp = ctx.request_object_pose(object_name)
        if not pose_resp or not pose_resp.success:
            return False, "Perception failed", 0.0, PoseStamped()
        
        confidence = float(pose_resp.confidence)
        if confidence < PICK_CONF_TH:
            return False, f"Low confidence {confidence:.2f}", confidence, PoseStamped()
        
        cam_pose = pose_resp.pose

    # ------------------------------------------------------------------
    # 공통: 좌표 변환 및 실행
    # ------------------------------------------------------------------
    
    # 1. TCP 위치 계산 (그리퍼 오프셋 적용)
    # RealSense 기준 Z축을 줄여서 그리퍼 끝이 물체 위치에 오게 함
    tcp_cam_pose = PoseStamped()
    tcp_cam_pose.header = cam_pose.header
    tcp_cam_pose.pose.position.x = cam_pose.pose.position.x
    tcp_cam_pose.pose.position.y = cam_pose.pose.position.y
    tcp_cam_pose.pose.position.z = cam_pose.pose.position.z - GRIPPER_OFFSET
    tcp_cam_pose.pose.orientation = cam_pose.pose.orientation

    if tcp_cam_pose.pose.position.z <= 0.05:
        # 안전장치: 계산된 Z 높이가 너무 낮거나 음수면 실행 중단
        return False, "Calculated Z is too close/negative", confidence, PoseStamped()

    # 2. Camera -> Base 좌표 변환
    base_xyz = ctx.transform_camera_to_base(tcp_cam_pose)
    bx, by, bz = base_xyz

    # 3. 모션 실행
    try:
        execute_pick_motion(ctx, bx, by, bz, target_yaw_rad=grasp_angle_rad)
        return True, "Success", confidence, ctx.make_final_pose(bx, by, bz)
    except Exception as e:
        ctx.node.get_logger().error(f"[PICK] Motion Error: {e}")
        return False, f"Motion Error: {e}", confidence, PoseStamped()
