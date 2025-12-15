from __future__ import annotations
from typing import Any, Dict, List
import time
import threading
import copy

_LOCK = threading.Lock()

_STATE: Dict[str, Any] = {
    "camera_ok": None,
    "last_frame_ts": None,

    "queue_size": 0,
    "queue_dropped": 0,

    "yolo": {
        "cls": None,
        "conf": None,
        "center": None,
        "bbox": None,
        "in_table_roi": None,
        "between_holes": None,
        "confirmed": [],   # ✅ 멀티 물체 리스트
    },

    "gpt_inference": False,
    "gpt_inference_since": None,
    "gpt_last_done_ts": None,

    "gpt": {
        "scene_summary": None,
        "risk_level": None,
    },
    "action": {
        "recommended_action": None,
    },
}

_EVENTS: List[Dict[str, Any]] = []
_EVENTS_MAX = 200


def _deep_merge(dst: Dict[str, Any], src: Dict[str, Any]) -> None:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_merge(dst[k], v)
        else:
            dst[k] = v


def update_state(patch: Dict[str, Any]) -> None:
    with _LOCK:
        _deep_merge(_STATE, patch)


def push_event(tag: str, msg: str) -> None:
    ev = {"t": time.time(), "tag": str(tag), "msg": str(msg)}
    with _LOCK:
        _EVENTS.insert(0, ev)
        if len(_EVENTS) > _EVENTS_MAX:
            del _EVENTS[_EVENTS_MAX:]


def get_state_snapshot() -> Dict[str, Any]:
    with _LOCK:
        snap = copy.deepcopy(_STATE)
        snap["events"] = list(_EVENTS[:50])
    return snap
