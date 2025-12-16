#!/bin/bash

# ============================================================
# Real Robot YOLO Pick & Place 시작 스크립트
# ============================================================
# 사용법: ./start_real_robot_yolo.sh
# 또는: bash start_real_robot_yolo.sh
# ============================================================

set -e  # 에러 발생 시 종료

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "╔════════════════════════════════════════════════════════╗"
echo "║  Real Robot YOLO Pick & Place Startup Script          ║"
echo "╚════════════════════════════════════════════════════════╝"
echo ""

# Step 1: 로봇 연결 대기
echo "📡 [1/3] Real Robot에 연결 중..."
echo "명령어 실행: real-robot"
echo ""

if command -v real-robot &> /dev/null; then
    real-robot
    echo "✅ Real Robot 연결 완료!"
else
    echo "⚠️  Warning: real-robot 명령어를 찾을 수 없습니다."
    echo "시스템에 real-robot이 설치되어 있는지 확인하세요."
    read -p "계속하시겠습니까? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo ""

# Step 2: ROS2 워크스페이스 소스 설정
echo "🔧 [2/3] ROS2 환경 설정 중..."
if [ -f "ros2_ws/install/setup.bash" ]; then
    source ros2_ws/install/setup.bash
    echo "✅ ROS2 환경 설정 완료!"
else
    echo "⚠️  Warning: ros2_ws/install/setup.bash를 찾을 수 없습니다."
    echo "먼저 다음 명령어를 실행하세요:"
    echo "  cd ros2_ws && colcon build"
    exit 1
fi

echo ""

# Step 3: YOLO PickPlace 실행
echo "🚀 [3/3] YOLO PickPlace 프로그램 시작..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "조작 방법:"
echo "  - 'p' 키: 감지된 물체를 집어서 옮기기 (pick & drop)"
echo "  - 'ESC' 키: 프로그램 종료"
echo ""
echo "RViz 실시간 시각화 (선택사항, 다른 터미널에서 실행):"
echo "  rviz2 -d /home/rokey/DUM-E/rviz_yolo_config.rviz"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# YOLO 프로그램 실행
python3 yolo_pickplace.py

echo ""
echo "╔════════════════════════════════════════════════════════╗"
echo "║            프로그램 종료됨                              ║"
echo "╚════════════════════════════════════════════════════════╝"
