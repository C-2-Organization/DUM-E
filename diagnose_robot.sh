#!/bin/bash

# ============================================================
# 로봇 연결 상태 진단 스크립트
# ============================================================
# 사용법: ./diagnose_robot.sh
# 또는: bash diagnose_robot.sh
# ============================================================

BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}"
echo "╔════════════════════════════════════════════════════════╗"
echo "║       로봇 연결 상태 진단 도구                         ║"
echo "╚════════════════════════════════════════════════════════╝"
echo -e "${NC}"
echo ""

# 1. ROS2 설치 확인
echo -e "${YELLOW}[1/5] ROS2 설치 확인${NC}"
if command -v ros2 &> /dev/null; then
    echo -e "${GREEN}✅ ROS2 설치됨${NC}"
    ros2 --version
else
    echo -e "${RED}❌ ROS2 설치 안 됨${NC}"
fi
echo ""

# 2. ROS2 환경 확인
echo -e "${YELLOW}[2/5] ROS2 환경 설정 확인${NC}"
if [ -f "/home/rokey/DUM-E/ros2_ws/install/setup.bash" ]; then
    echo -e "${GREEN}✅ ros2_ws/install/setup.bash 존재${NC}"
    source /home/rokey/DUM-E/ros2_ws/install/setup.bash
    echo "   ROS_DISTRO: ${ROS_DISTRO:-설정안됨}"
else
    echo -e "${RED}❌ ros2_ws/install/setup.bash 없음${NC}"
    echo "   해결: cd /home/rokey/DUM-E/ros2_ws && colcon build"
fi
echo ""

# 3. 네트워크 연결 상태
echo -e "${YELLOW}[3/5] 네트워크 연결 상태${NC}"
if ping -c 1 8.8.8.8 &> /dev/null; then
    echo -e "${GREEN}✅ 인터넷 연결됨${NC}"
else
    echo -e "${RED}⚠️  인터넷 연결 없음${NC}"
fi

# 로봇 IP 핑 테스트
echo ""
echo "로봇 연결 테스트:"
ROBOT_IPS=("192.168.1.100" "192.168.1.101" "192.168.1.1" "127.0.0.1")
FOUND_ROBOT=0

for ip in "${ROBOT_IPS[@]}"; do
    if ping -c 1 -W 1 $ip &> /dev/null; then
        echo -e "  ${GREEN}✅ $ip (응답함)${NC}"
        FOUND_ROBOT=1
    else
        echo -e "  ${RED}❌ $ip (응답 없음)${NC}"
    fi
done

if [ $FOUND_ROBOT -eq 0 ]; then
    echo -e "  ${YELLOW}⚠️  모든 기본 IP에서 응답 없음${NC}"
    echo "     직접 IP 확인:"
    echo "     $ nmap -p 12345 192.168.1.0/24"
fi
echo ""

# 4. 네트워크 인터페이스 확인
echo -e "${YELLOW}[4/5] 네트워크 인터페이스${NC}"
echo "활성 네트워크:"
ip -4 addr show | grep -E "inet " | grep -v "127.0.0.1"
echo ""

# 5. ROS2 노드 상태
echo -e "${YELLOW}[5/5] ROS2 노드 상태${NC}"

echo "현재 실행 중인 노드:"
ros2 node list 2>&1 | head -10

echo ""
echo "로봇 관련 노드:"
if ros2 node list 2>&1 | grep -q "dsr"; then
    echo -e "${GREEN}✅ 로봇 노드 감지됨${NC}"
    ros2 node list | grep dsr
else
    echo -e "${RED}❌ 로봇 노드 없음 (로봇 연결 필요)${NC}"
fi

echo ""
echo "카메라 관련 노드:"
if ros2 node list 2>&1 | grep -q "camera"; then
    echo -e "${GREEN}✅ 카메라 노드 감지됨${NC}"
    ros2 node list | grep camera
else
    echo -e "${RED}⚠️  카메라 노드 없음${NC}"
fi

echo ""
echo "════════════════════════════════════════════════════════"
echo ""

# 권장사항
echo -e "${YELLOW}📋 권장 조치:${NC}"
echo ""

if ! ros2 node list 2>&1 | grep -q "dsr"; then
    echo "1️⃣  로봇 연결 필요:"
    echo "   cd /home/rokey/DUM-E"
    echo "   ./connect_real_robot.sh"
    echo ""
fi

if ! ros2 node list 2>&1 | grep -q "camera"; then
    echo "2️⃣  카메라 시작 필요:"
    echo "   ros2 launch realsense2_camera rs_align_depth_launch.py"
    echo ""
fi

echo "3️⃣  ROS2 노드 모니터링:"
echo "   ros2 topic list"
echo "   ros2 node list"
echo ""

echo "4️⃣  YOLO 프로그램 실행:"
echo "   cd /home/rokey/DUM-E"
echo "   python3 yolo_pickplace.py"
echo ""

echo -e "${GREEN}진단 완료!${NC}"
