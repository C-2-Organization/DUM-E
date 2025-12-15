# dum_e_perception_bridge/lookat_map.py
LOOKAT_PRESETS = {
    "10~11번 사이에 있습니다": {"x": 0.40, "y": -0.18, "z": 0.35, "rx": 180, "ry": 0, "rz": 90},
    "11~12번 사이에 있습니다": {"x": 0.42, "y": -0.20, "z": 0.35, "rx": 180, "ry": 0, "rz": 90},
    # 필요 영역 더 추가
}

def get_lookat_pose(between_holes: str):
    return LOOKAT_PRESETS.get(between_holes)
