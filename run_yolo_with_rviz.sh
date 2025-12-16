#!/bin/bash

# ============================================================
# YOLO + RViz 동시 실행 스크립트
# ============================================================
# 카메라 frame을 먼저 생성한 후 YOLO와 RViz를 함께 시작

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}"
echo "╔════════════════════════════════════════════════════════╗"
echo "║         YOLO + RViz Synchronized Startup              ║"
echo "╚════════════════════════════════════════════════════════╝"
echo -e "${NC}"
echo ""

# ROS2 환경 설정
echo -e "${YELLOW}🔧 ROS2 환경 설정 중...${NC}"
source ros2_ws/install/setup.bash
export ROS_DOMAIN_ID=0

echo -e "${GREEN}✅ ROS2 환경 설정 완료!${NC}"
echo ""

# Step 1: 카메라 노드 확인 (frame이 필요하기 때문)
echo -e "${YELLOW}📷 카메라 노드 확인 중...${NC}"
sleep 1

if ros2 node list 2>/dev/null | grep -q "camera"; then
    echo -e "${GREEN}✅ 카메라 이미 실행 중${NC}"
else
    echo -e "${YELLOW}⚠️  카메라를 별도로 시작하세요:${NC}"
    echo "   ros2 launch realsense2_camera rs_align_depth_launch.py"
fi

echo ""

# Step 2: RViz 시작
echo -e "${YELLOW}🖼️  RViz 시작 중...${NC}"
rviz2 -d rviz_yolo_config.rviz 2>/dev/null &
RVIZ_PID=$!
sleep 2

echo -e "${GREEN}✅ RViz 시작됨 (PID: $RVIZ_PID)${NC}"
echo ""

# Step 3: YOLO 시작
echo -e "${YELLOW}🚀 YOLO PickPlace 시작...${NC}"
echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║              프로그램 실행 중                          ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════╝${NC}"
echo ""

python3 yolo_pickplace.py

# YOLO 종료 시 RViz도 종료
echo ""
echo -e "${YELLOW}YOLO 프로그램 종료...${NC}"
kill $RVIZ_PID 2>/dev/null || true

echo -e "${GREEN}완료!${NC}"
