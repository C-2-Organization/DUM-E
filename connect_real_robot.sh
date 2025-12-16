#!/bin/bash

# ============================================================
# Real Robot (실제 로봇) 연결 스크립트
# ============================================================
# 사용법: ./connect_real_robot.sh [host] [port] [model]
# 예시: ./connect_real_robot.sh 192.168.1.100 12345 m0609
# ============================================================

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# 기본값
ROBOT_HOST="${1:-192.168.1.100}"
ROBOT_PORT="${2:-12345}"
ROBOT_MODEL="${3:-m0609}"

# 색상
BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}"
echo "╔════════════════════════════════════════════════════════╗"
echo "║         Real Robot Connection Script                   ║"
echo "╚════════════════════════════════════════════════════════╝"
echo -e "${NC}"
echo ""

echo "📡 연결 설정:"
echo "  Host:  $ROBOT_HOST"
echo "  Port:  $ROBOT_PORT"
echo "  Model: $ROBOT_MODEL"
echo ""

# ROS2 환경 설정
echo -e "${YELLOW}🔧 ROS2 환경 설정 중...${NC}"
if [ ! -f "ros2_ws/install/setup.bash" ]; then
    echo -e "${RED}❌ Error: ros2_ws/install/setup.bash를 찾을 수 없습니다.${NC}"
    echo "먼저 다음 명령어를 실행하세요:"
    echo "  cd ros2_ws && colcon build"
    exit 1
fi

source ros2_ws/install/setup.bash
export ROS_DOMAIN_ID=0

echo -e "${GREEN}✅ ROS2 환경 설정 완료!${NC}"
echo ""

# 로봇 연결
echo -e "${YELLOW}📡 실제 로봇에 연결 중...${NC}"
echo ""
echo "명령어:"
echo "  ros2 launch dsr_bringup2 dsr_bringup2_rviz.launch.py mode:=real host:=$ROBOT_HOST port:=$ROBOT_PORT model:=$ROBOT_MODEL"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

ros2 launch dsr_bringup2 dsr_bringup2_rviz.launch.py \
    mode:=real \
    host:=$ROBOT_HOST \
    port:=$ROBOT_PORT \
    model:=$ROBOT_MODEL &
BRINGUP_PID=$!

sleep 3

# YOLO + 마커 시작 (카메라 프레임에서 감지된 물체 표시)
echo -e "${YELLOW}🚀 YOLO PickPlace 시작...${NC}"
echo ""

python3 yolo_pickplace.py

# YOLO 종료 시 로봇 연결도 종료
kill $BRINGUP_PID 2>/dev/null || true

echo ""
echo -e "${GREEN}✅ 로봇 연결 종료${NC}"
