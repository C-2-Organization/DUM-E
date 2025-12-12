# webcam/services/action_dispatcher.py

def dispatch(info: dict):
    """
    GPT에서 온 상황 JSON(info)을 보고
    로봇/시스템 액션을 결정하는 자리.

    지금은 ROS 연동 대신 print로만 표시.
    """
    action = info.get("recommended_action", "idle_monitor")
    summary = info.get("scene_summary", "")
    risk = info.get("risk_level", "low")

    print(f"[SITUATION] {summary} | risk={risk}")
    print(f"[ACTION] {action}")

    # 여기에 ROS2 서비스/토픽 호출 붙이면 됨
    if action == "clean_spill":
        print("[ROS] → clean_spill 시퀀스 실행")
    elif action == "pick_up_object":
        print("[ROS] → 물체 집기 시퀀스 실행")
    elif action == "look_at_direction":
        print("[ROS] → 해당 방향으로 바라보기")
    elif action == "handover_to_human":
        print("[ROS] → handover 시퀀스 실행")
    elif action == "avoid_human":
        print("[ROS] → 회피 경로 계획")
    else:
        # idle_monitor 포함
        print("[ROS] → 아무 행동 안 함 (모니터링)")
