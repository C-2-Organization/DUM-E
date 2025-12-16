# dum_e_webcam_bridge/webcam_gate.py
from __future__ import annotations

import os
import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import String

from dum_e_interfaces.srv import RunSkill
from dum_e_interfaces.msg import SkillCommand


# ---------------------------
# Utils
# ---------------------------
def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "y", "on")


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)))
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env(name, str(default)))
    except Exception:
        return default


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _get_lookat_skill_type() -> int:
    """
    안전하게 LOOK_AT skill_type 얻기:
    1) SkillCommand.LOOK_AT 있으면 그걸 사용
    2) 없으면 env LOOKAT_SKILL_TYPE 사용(최후 fallback)
    """
    if hasattr(SkillCommand, "LOOK_AT"):
        return int(getattr(SkillCommand, "LOOK_AT"))
    return _env_int("LOOKAT_SKILL_TYPE", 2)


# ---------------------------
# Config
# ---------------------------
@dataclass(frozen=True)
class GateConfig:
    # Topics / Services
    in_topic: str = "/dum_e/webcam/webcam"
    run_skill_service: str = "/run_skill"

    # Filtering
    min_conf: float = 0.20
    min_hit: int = 3
    require_miss0: bool = True
    prefer_table: bool = True

    # Stabilization
    lock_sec: float = 1.0
    call_min_interval: float = 0.7
    min_move_mm: float = 20.0

    # Look-at params
    off_x_mm: float = 0.0
    off_y_mm: float = 0.0

    # Logging
    debug: bool = False
    wait_service_sec: float = 5.0

    @staticmethod
    def from_env() -> "GateConfig":
        return GateConfig(
            in_topic=_env("WEBCAM_GATE_IN_TOPIC", "/dum_e/webcam/webcam"),
            run_skill_service=_env("WEBCAM_GATE_SERVICE", "/run_skill"),
            min_conf=_env_float("GATE_MIN_CONF", 0.20),
            min_hit=_env_int("GATE_MIN_HIT", 3),
            require_miss0=_env_bool("GATE_REQUIRE_MISS0", True),
            prefer_table=_env_bool("GATE_PREFER_TABLE", True),
            lock_sec=_env_float("GATE_LOCK_SEC", 1.0),
            call_min_interval=_env_float("GATE_CALL_MIN_INTERVAL", 0.7),
            min_move_mm=_env_float("GATE_MIN_MOVE_MM", 20.0),
            off_x_mm=_env_float("LOOK_OFFSET_X_MM", 0.0),
            off_y_mm=_env_float("LOOK_OFFSET_Y_MM", 0.0),
            debug=_env_bool("GATE_DEBUG", False),
            wait_service_sec=_env_float("WEBCAM_GATE_WAIT_SERVICE_SEC", 5.0),
        )


# ---------------------------
# Node
# ---------------------------
class WebcamGate(Node):
    """
    IN_TOPIC (JSON String)을 받아서:
      - candidates 필터/랭킹/락/스로틀 적용
      - best 1개만 뽑아서 /run_skill(LOOK_AT) 호출
    """

    def __init__(self):
        super().__init__("webcam_gate")

        self.cfg = GateConfig.from_env()
        self.lookat_skill_type = _get_lookat_skill_type()

        self.create_subscription(String, self.cfg.in_topic, self.cb, qos_profile_sensor_data)
        self.cli = self.create_client(RunSkill, self.cfg.run_skill_service)

        self.get_logger().info(f"Waiting for {self.cfg.run_skill_service} service... ({self.cfg.wait_service_sec}s)")
        self.cli.wait_for_service(timeout_sec=self.cfg.wait_service_sec)

        # ---- state ----
        self._locked_track_id: Optional[int] = None
        self._lock_until_ts: float = 0.0
        self._last_call_ts: float = 0.0
        self._last_sent_xy: Optional[Tuple[float, float]] = None
        self._busy = False
        self._busy_until = 0.0

        self.get_logger().info(
            "WebcamGate ready | "
            f"in_topic='{self.cfg.in_topic}' "
            f"LOOK_AT skill_type={self.lookat_skill_type} "
            f"min_conf={self.cfg.min_conf} min_hit={self.cfg.min_hit} miss0={self.cfg.require_miss0} "
            f"prefer_table={self.cfg.prefer_table} "
            f"lock={self.cfg.lock_sec}s call_interval={self.cfg.call_min_interval}s min_move={self.cfg.min_move_mm}mm "
            f"debug={self.cfg.debug}"
        )

    # ---------------------------
    # Candidate selection helpers
    # ---------------------------
    def _score(self, c: Dict[str, Any]) -> float:
        conf = float(c.get("conf") or 0.0)
        hit = float(c.get("hit") or 0.0)
        return 0.7 * conf + 0.3 * _clamp01(hit / 10.0)

    @staticmethod
    def _dist_mm(a: Tuple[float, float], b: Tuple[float, float]) -> float:
        dx = a[0] - b[0]
        dy = a[1] - b[1]
        return (dx * dx + dy * dy) ** 0.5

    def _filter_candidates(self, cands: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for c in cands:
            xy = c.get("robot_xy")
            if not (isinstance(xy, (list, tuple)) and len(xy) >= 2):
                continue

            conf = float(c.get("conf") or 0.0)
            hit = int(c.get("hit") or 0)
            miss = int(c.get("miss") or 0)

            if conf < self.cfg.min_conf:
                continue
            if hit < self.cfg.min_hit:
                continue
            if self.cfg.require_miss0 and miss != 0:
                continue

            out.append(c)
        return out

    def _pick_best(self, cands: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not cands:
            return None

        pool = cands
        if self.cfg.prefer_table:
            table = [c for c in cands if c.get("in_table_roi") is True]
            if table:
                pool = table

        pool = sorted(
            pool,
            key=lambda c: (self._score(c), float(c.get("conf") or 0.0)),
            reverse=True,
        )
        return pool[0] if pool else None

    @staticmethod
    def _find_by_track(cands: List[Dict[str, Any]], tid: int) -> Optional[Dict[str, Any]]:
        for c in cands:
            if c.get("track_id") == tid:
                return c
        return None

    def _should_trigger_action(self, action_raw: Any) -> bool:
        action = (action_raw or "idle").strip()
        return action in ("look_at", "inspect", "track", "look")

    def _update_lock(self, now: float, best: Dict[str, Any]) -> None:
        tid = best.get("track_id")
        if tid is None:
            self._locked_track_id = None
            self._lock_until_ts = 0.0
            return
        self._locked_track_id = int(tid)
        self._lock_until_ts = now + self.cfg.lock_sec

    def _passes_min_move(self, best: Dict[str, Any]) -> bool:
        xy = best.get("robot_xy")
        try:
            xy2 = (float(xy[0]), float(xy[1]))
        except Exception:
            return False

        if self._last_sent_xy is not None:
            if self._dist_mm(self._last_sent_xy, xy2) < self.cfg.min_move_mm:
                return False

        self._last_sent_xy = xy2
        self._busy = False
        self._busy_until = 0.0

        return True

    # ---------------------------
    # ROS call
    # ---------------------------
    def _call_run_skill_lookat(self, best: Dict[str, Any], stamp: Any) -> None:
        if not self.cli.service_is_ready():
            self.get_logger().warn(f"'{self.cfg.run_skill_service}' service not ready")
            return

        xy = best.get("robot_xy")
        target_xy = None
        if isinstance(xy, (list, tuple)) and len(xy) >= 2:
            target_xy = [float(xy[0]), float(xy[1])]

        params = {
            # motion 쪽이 candidates를 찾는 경우 대비
            "candidates": [best],
            "best": best,

            # 어떤 구현은 target_xy만 보는 경우가 있음
            "target_xy": target_xy,

            "stamp": stamp,

            # look_at.py에서 offset만 쓰게 유지
            "offset_x": self.cfg.off_x_mm,
            "offset_y": self.cfg.off_y_mm,
        }

        req = RunSkill.Request()
        cmd = SkillCommand()
        cmd.skill_type = self.lookat_skill_type
        cmd.object_name = ""
        cmd.target_pose.header.frame_id = ""
        cmd.params_json = json.dumps(params, ensure_ascii=False)
        req.command = cmd

        self.get_logger().info(
            f"[GATE] -> /run_skill LOOK_AT | track_id={best.get('track_id')} "
            f"conf={best.get('conf')} hit={best.get('hit')} miss={best.get('miss')} robot_xy={best.get('robot_xy')}"
        )

        # 동작 시작: 응답 올 때까지 잠금(안전 타임아웃 포함)
        self._busy = True
        self._busy_until = time.time() + 3.0   # 3초 타임아웃(원하면 env로 빼도 됨)
        
        fut = self.cli.call_async(req)
        fut.add_done_callback(self._on_done)

    def _on_done(self, fut) -> None:
        try:
            resp = fut.result()
        except Exception as e:
            self.get_logger().error(f"/run_skill call failed: {e}")
            return

        self.get_logger().info(
            f"/run_skill resp: success={getattr(resp,'success',None)} "
            f"message={getattr(resp,'message',None)} "
            f"confidence={getattr(resp,'confidence',None)}"
        )
        
        self._busy = False
        self._busy_until = 0.0

    # ---------------------------
    # Callback
    # ---------------------------
    def cb(self, msg: String) -> None:
        now = time.time()

        # throttle
        if (now - self._last_call_ts) < self.cfg.call_min_interval:
            return

        if self.cfg.debug:
            self.get_logger().info(f"[GATE] RX len={len(msg.data)}")
            
        # busy: 이전 run_skill 응답 오기 전에는 새 호출 막기
        if self._busy and now < self._busy_until:
            return

        # parse JSON
        try:
            data = json.loads(msg.data)
        except Exception as e:
            self.get_logger().warn(f"[GATE] JSON parse fail: {e}")
            return

        action_raw = data.get("recommended_action")
        if not self._should_trigger_action(action_raw):
            return

        stamp = data.get("stamp")
        cands_raw = data.get("candidates") or []
        if self.cfg.debug:
            self.get_logger().info(f"[GATE] action={action_raw} candidates={len(cands_raw)}")

        # filter
        filtered = self._filter_candidates(cands_raw)
        if not filtered:
            # lock clear if no viable candidates
            self._locked_track_id = None
            self._lock_until_ts = 0.0
            return

        # lock 유지
        best: Optional[Dict[str, Any]] = None
        if self._locked_track_id is not None and now < self._lock_until_ts:
            locked = self._find_by_track(filtered, self._locked_track_id)
            if locked is not None:
                best = locked

        # 새로 선택
        if best is None:
            best = self._pick_best(filtered)
            if best is None:
                return
            self._update_lock(now, best)

        # min move
        if not self._passes_min_move(best):
            return

        # call
        self._call_run_skill_lookat(best, stamp)
        self._last_call_ts = now


def main():
    rclpy.init()
    node = WebcamGate()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
