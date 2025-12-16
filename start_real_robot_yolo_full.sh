#!/bin/bash

# ============================================================
# Real Robot YOLO Pick & Place 종합 시작 스크립트 (모든 것 한 번에)
# ============================================================
# 사용법: ./start_real_robot_yolo_full.sh
# 또는: bash start_real_robot_yolo_full.sh
#
# 이 스크립트는 다음을 자동으로 실행합니다:
# 1. Real Robot 연결
# 2. ROS2 환경 설정
# 3. 여러 터미널/프로세스 관리
# ============================================================

set -e  # 에러 발생 시 종료

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}"
echo "╔════════════════════════════════════════════════════════╗"
echo "║  Real Robot YOLO Pick & Place Complete Startup        ║"
echo "║     (로봇 + RViz + YOLO 동시 실행)                    ║"
echo "╚════════════════════════════════════════════════════════╝"
echo -e "${NC}"
echo ""

# Step 1: 로봇 연결
echo -e "${YELLOW}📡 [Step 1/3] Real Robot에 연결 중...${NC}"
echo "명령어 실행: real-robot"
echo ""

if command -v real-robot &> /dev/null; then
    real-robot
    echo -e "${GREEN}✅ Real Robot 연결 완료!${NC}"
else
    echo -e "${RED}⚠️  Warning: real-robot 명령어를 찾을 수 없습니다.${NC}"
    echo "시스템에 real-robot이 설치되어 있는지 확인하세요."
    read -p "계속하시겠습니까? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo ""

# Step 2: ROS2 환경 설정
echo -e "${YELLOW}🔧 [Step 2/3] ROS2 환경 설정 중...${NC}"
if [ -f "ros2_ws/install/setup.bash" ]; then
    source ros2_ws/install/setup.bash
    export ROS_DOMAIN_ID=0
    echo -e "${GREEN}✅ ROS2 환경 설정 완료!${NC}"
else
    echo -e "${RED}⚠️  Error: ros2_ws/install/setup.bash를 찾을 수 없습니다.${NC}"
    echo "먼저 다음 명령어를 실행하세요:"
    echo "  cd ros2_ws && colcon build"
    exit 1
fi

echo ""

# Step 3: 백그라운드 프로세스 시작
echo -e "${YELLOW}🚀 [Step 3/3] 모든 프로세스 시작 중...${NC}"
echo ""

# YOLO 프로세스를 백그라운드에서 실행
echo -e "${BLUE}→ YOLO PickPlace 시작...${NC}"
python3 yolo_pickplace.py &
YOLO_PID=$!
echo "  PID: $YOLO_PID"

# 선택: RViz도 함께 실행할지 묻기
echo ""
read -p "RViz도 함께 실행하시겠습니까? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${BLUE}→ RViz 시작...${NC}"
    rviz2 -d "$SCRIPT_DIR/rviz_yolo_config.rviz" 2>/dev/null &
    RVIZ_PID=$!
    echo "  PID: $RVIZ_PID"
fi

echo ""
echo -e "${GREEN}════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}✅ 모든 프로세스가 시작되었습니다!${NC}"
echo -e "${GREEN}════════════════════════════════════════════════════════${NC}"
echo ""

echo "📋 실행 중인 프로세스:"
echo "  - YOLO PickPlace (PID: $YOLO_PID)"
if [ ! -z "$RVIZ_PID" ]; then
    echo "  - RViz (PID: $RVIZ_PID)"
fi
echo ""

echo "📝 조작 방법:"
echo "  - YOLO 윈도우에서 'p' 키: 감지된 물체 집기"
echo "  - YOLO 윈도우에서 'ESC' 키: 프로그램 종료"
echo ""

echo "⏹️  종료 방법:"
echo "  - YOLO 윈도우를 닫거나 ESC를 눌러주세요"
echo "  - 또는 터미널에서 Ctrl+C를 눌러주세요"
echo ""

# 모든 백그라운드 프로세스가 끝날 때까지 대기
wait $YOLO_PID 2>/dev/null
YOLO_EXIT=$?

echo ""
echo -e "${BLUE}════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}YOLO PickPlace 종료됨 (Exit Code: $YOLO_EXIT)${NC}"

if [ ! -z "$RVIZ_PID" ]; then
    kill $RVIZ_PID 2>/dev/null || true
    echo -e "${BLUE}RViz 종료됨${NC}"
fi

echo -e "${BLUE}════════════════════════════════════════════════════════${NC}"
