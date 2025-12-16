#!/bin/bash

# ============================================================
# YOLO PickPlace 간단 실행 스크립트
# ============================================================
# 사용법: ./run_yolo_simple.sh
# 카메라만 있으면 바로 실행 가능!
# ============================================================

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# 색상
BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}"
echo "╔════════════════════════════════════════════════════════╗"
echo "║         YOLO Object Detection PickPlace               ║"
echo "║        (카메라만으로 실행 - 로봇 연결 불필요)         ║"
echo "╚════════════════════════════════════════════════════════╝"
echo -e "${NC}"
echo ""

# ROS2 환경 설정
echo -e "${YELLOW}🔧 ROS2 환경 설정 중...${NC}"
if [ ! -f "ros2_ws/install/setup.bash" ]; then
    echo -e "❌ ros2_ws/install/setup.bash를 찾을 수 없습니다."
    echo "먼저 다음을 실행하세요:"
    echo "  cd ros2_ws && colcon build"
    exit 1
fi

source ros2_ws/install/setup.bash
export ROS_DOMAIN_ID=0

echo -e "${GREEN}✅ ROS2 환경 설정 완료!${NC}"
echo ""

# YOLO 프로그램 실행
echo -e "${YELLOW}🚀 YOLO PickPlace 시작...${NC}"
echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║              프로그램 실행 중                          ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════╝${NC}"
echo ""
echo "📋 화면 조작:"
echo "  - 'p' 키: Pick & Place 실행 (로봇 연결 필요)"
echo "  - 'ESC' 키: 프로그램 종료"
echo ""
echo "🎯 감지 결과:"
echo "  - OpenCV 윈도우에 바운딩 박스 표시"
echo "  - RViz에 3D 마커 표시"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

python3 yolo_pickplace.py

echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║              프로그램 종료됨                          ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════╝${NC}"
