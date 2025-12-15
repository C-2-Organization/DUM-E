# dum_e_webcam_bridge/webcam_gate.py
from __future__ import annotations

import os
import json
import time
from typing import Any, Dict, List, Optional, Tuple

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from dum_e_interfaces.srv import RunSkill
from dum_e_interfaces.msg import SkillCommand

from rclpy.qos import qos_profile_sensor_data


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
    return int(os.getenv("LOOKAT_SKILL_TYPE", "2"))


def _topic(name: str, default: str) -> str:
    """토픽명도 하드코딩 줄이기: env 우선"""
    return os.getenv(name, default)


class WebcamGate(Node):
    """
    IN_TOPIC (JSON String)을 받아서:
      - candidates 필터/랭킹/락/스로틀 적용
      - best 1개만 뽑아서 /run_skill(LOOK_AT) 호출
    """

    def __init__(self):
        super().__init__("webcam_gate")

        # ✅ 토픽 하드코딩 제거 (env로 제어)
        self.in_topic = _topic("WEBCAM_GATE_IN_TOPIC", "/dum_e/webcam/webcam")

        # ✅ 센서 스트림/비정기 스트링에도 잘 맞는 QoS
        self.create_subscription(String, self.in_topic, self.cb, qos_profile_sensor_data)

        self.cli = self.create_client(RunSkill, "/run_skill")

        # ✅ 서비스 대기(너 receiver처럼 안전하게)
        wait_sec = float(os.getenv("WEBCAM_GATE_WAIT_SERVICE_SEC", "5.0"))
        self.get_logger().info(f"Waiting for /run_skill service... ({wait_sec}s)")
        self.cli.wait_for_service(timeout_sec=wait_sec)

        # ✅ 스킬 타입: enum 우선(하드코딩 X)
        self.lookat_skill_type = _get_lookat_skill_type()

        # ---- 튜닝 파라미터 (env로 조절 가능) ----
        self.min_conf = float(os.getenv("GATE_MIN_CONF", "0.20"))
        self.min_hit = int(os.getenv("GATE_MIN_HIT", "3"))
        self.require_miss0 = os.getenv("GATE_REQUIRE_MISS0", "1").lower() in ("1", "true", "yes")
        self.prefer_table = os.getenv("GATE_PREFER_TABLE", "1").lower() in ("1", "true", "yes")

        self.lock_sec = float(os.getenv("GATE_LOCK_SEC", "1.0"))
        self.call_min_interval = float(os.getenv("GATE_CALL_MIN_INTERVAL", "0.7"))
        self.min_move_mm = float(os.getenv("GATE_MIN_MOVE_MM", "20"))

        # look_at 기본 파라미터(mm/deg)
        self.look_z_mm = float(os.getenv("LOOK_Z_MM", "350"))
        self.look_rx = float(os.getenv("LOOK_RX", "180"))
        self.look_ry = float(os.getenv("LOOK_RY", "0"))
        self.look_rz = float(os.getenv("LOOK_RZ", "90"))
        self.off_x_mm = float(os.getenv("LOOK_OFFSET_X_MM", "0"))
        self.off_y_mm = float(os.getenv("LOOK_OFFSET_Y_MM", "0"))

        # ---- 내부 상태(락/스로틀) ----
        self._locked_track_id: Optional[int] = None
        self._lock_until_ts: float = 0.0
        self._last_call_ts: float = 0.0
        self._last_sent_xy: Optional[Tuple[float, float]] = None

        # ✅ 여기서는 data 같은 런타임 변수 쓰면 안 됨 (cb에서만 있음)
        self.get_logger().info(
            "WebcamGate ready | "
            f"in_topic='{self.in_topic}' "
            f"LOOK_AT skill_type={self.lookat_skill_type} "
            f"min_conf={self.min_conf} min_hit={self.min_hit} miss0={self.require_miss0} "
            f"prefer_table={self.prefer_table} "
            f"lock={self.lock_sec}s call_interval={self.call_min_interval}s min_move={self.min_move_mm}mm"
        )

    # ---------- 선택 로직 ----------
    def _score(self, c: Dict[str, Any]) -> float:
        conf = float(c.get("conf") or 0.0)
        hit = float(c.get("hit") or 0.0)
        return 0.7 * conf + 0.3 * _clamp01(hit / 10.0)

    def _dist_mm(self, a: Tuple[float, float], b: Tuple[float, float]) -> float:
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

            if conf < self.min_conf:
                continue
            if hit < self.min_hit:
                continue
            if self.require_miss0 and miss != 0:
                continue

            out.append(c)
        return out

    def _pick_best(self, cands: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not cands:
            return None

        if self.prefer_table:
            table = [c for c in cands if c.get("in_table_roi") is True]
            if table:
                cands = table

        cands = sorted(
            cands,
            key=lambda c: (self._score(c), float(c.get("conf") or 0.0)),
            reverse=True,
        )
        return cands[0] if cands else None

    def _find_by_track(self, cands: List[Dict[str, Any]], tid: int) -> Optional[Dict[str, Any]]:
        for c in cands:
            if c.get("track_id") == tid:
                return c
        return None

    # ---------- ROS 콜 ----------
    def _call_run_skill_lookat(self, best: Dict[str, Any], stamp: Any):
        if not self.cli.service_is_ready():
            self.get_logger().warn("'/run_skill' service not ready")
            return

        params = {
            "best": best,
            "stamp": stamp,
            "z_mm": self.look_z_mm,
            "rx": self.look_rx,
            "ry": self.look_ry,
            "rz": self.look_rz,
            "offset_x": self.off_x_mm,
            "offset_y": self.off_y_mm,
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

        fut = self.cli.call_async(req)
        fut.add_done_callback(self._on_done)

    def _on_done(self, fut):
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

    # ---------- callback ----------
    def cb(self, msg: String):
        # 1) 수신 확인
        self.get_logger().info(f"[GATE] RX len={len(msg.data)}")

        now = time.time()

        # 2) throttle
        if (now - self._last_call_ts) < self.call_min_interval:
            return

        # 3) JSON 파싱
        try:
            data = json.loads(msg.data)
        except Exception as e:
            self.get_logger().warn(f"[GATE] JSON parse fail: {e}")
            return

        # 4) 핵심 필드 로깅
        action_raw = data.get("recommended_action")
        cands_raw = data.get("candidates")
        self.get_logger().info(
            f"[GATE] action={action_raw} candidates={len(cands_raw or [])}"
        )

        action = (action_raw or "idle").strip()
        stamp = data.get("stamp")
        cands = cands_raw or []

        # 5) look_at 계열만 실행
        if action not in ("look_at", "inspect", "track", "look"):
            return

        # 6) candidates 필터
        filtered = self._filter_candidates(cands)
        if not filtered:
            if self._locked_track_id is not None:
                self._locked_track_id = None
                self._lock_until_ts = 0.0
            return

        # 7) 락 유지
        best = None
        if self._locked_track_id is not None and now < self._lock_until_ts:
            locked = self._find_by_track(filtered, self._locked_track_id)
            if locked is not None:
                best = locked

        # 8) 새로 선택
        if best is None:
            best = self._pick_best(filtered)
            if best is None:
                return
            tid = best.get("track_id")
            if tid is not None:
                self._locked_track_id = int(tid)
                self._lock_until_ts = now + self.lock_sec

        # 9) min_move_mm 체크
        xy = best.get("robot_xy")
        xy2 = (float(xy[0]), float(xy[1]))
        if self._last_sent_xy is not None:
            if self._dist_mm(self._last_sent_xy, xy2) < self.min_move_mm:
                return
        self._last_sent_xy = xy2

        # 10) call
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
