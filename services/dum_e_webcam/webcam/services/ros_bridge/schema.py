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
        "stamp": time.time(),
        "source": "webcam",
        "recommended_action": recommended_action,
        "risk_level": risk_level,
        "human_present": human_present,
        "hand_near_target": hand_near_target,
        "candidates": candidates,  # track_id/conf/hit/center/bbox/in_table_roi/between_holes ...
    }
