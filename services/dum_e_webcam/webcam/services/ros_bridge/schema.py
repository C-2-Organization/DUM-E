# webcam/services/ros_bridge/schema.py
from __future__ import annotations
from typing import Any, Dict, List, Optional
import time

def build_perception_msg(
    candidates: List[Dict[str, Any]],
    recommended_action: str = "look_at",
    risk_level: str = "low",
    human_present: bool = False,
    hand_near_target: bool = False,
) -> Dict[str, Any]:
    return {
        "stamp": 123.45,
        "source": "webcam",
        "recommended_action": "look_at",
        "best": {
            "track_id": 7,
            "cls_name": "tool",
            "conf": 0.31,
            "hit": 12,
            "robot_xy": [402.3, -180.1],
            "between_holes": "4~8번 사이에 있습니다"
        },
        "candidates": [...] 
    }
