# 로봇 연결 상태 확인 및 해결 가이드

## 🔍 현재 상황 분석

시스템 확인 결과:
- ✅ **RealSense 카메라**: 실행 중 (카메라 노드 활성화)
- ❌ **로봇 노드**: 미연결 (dsr_* 노드가 없음)
- ❌ **로봇 제어 노드**: 미실행

---

## 🚀 로봇 연결하기 (3가지 방법)

### 방법 1️⃣: 자동 연결 스크립트 (추천)

```bash
cd /home/rokey/DUM-E
./connect_real_robot.sh [HOST] [PORT] [MODEL]
```

**예시:**
```bash
# 기본값 사용 (192.168.1.100:12345, m0609 모델)
./connect_real_robot.sh

# 커스텀 설정
./connect_real_robot.sh 192.168.1.100 12345 m0609
```

**파라미터:**
- `HOST`: 로봇 IP 주소 (기본값: 192.168.1.100)
- `PORT`: 로봇 포트 (기본값: 12345)
- `MODEL`: 로봇 모델 (기본값: m0609)

---

### 방법 2️⃣: 수동 연결 (ROS2 명령어)

```bash
# 1. 터미널을 열고 ROS2 환경 설정
cd /home/rokey/DUM-E
source ros2_ws/install/setup.bash

# 2. 로봇 런치 파일 실행
ros2 launch dsr_bringup2 dsr_bringup2_rviz.launch.py \
    mode:=real \
    host:=192.168.1.100 \
    port:=12345 \
    model:=m0609
```

---

### 방법 3️⃣: 가상 모드 (테스트 용도)

실제 로봇이 없거나 연결할 수 없을 때:

```bash
cd /home/rokey/DUM-E
source ros2_ws/install/setup.bash

ros2 launch dsr_bringup2 dsr_bringup2_rviz.launch.py \
    mode:=virtual \
    host:=127.0.0.1 \
    port:=12345 \
    model:=m0609
```

---

## ⚙️ 로봇 설정값 확인

파일: `/home/rokey/DUM-E/yolo_pickplace.py`

현재 설정:
```python
ROBOT_ID = "dsr01"           # 로봇 ID
GRIPPER_NAME = "rg2"         # 그리퍼 이름
TOOLCHARGER_IP = "192.168.1.1"  # 툴 차저 IP
TOOLCHARGER_PORT = 502       # 툴 차저 포트
```

**필요 시 수정:**
```bash
nano yolo_pickplace.py
```

---

## ✅ 연결 확인

### 1️⃣ ROS2 노드 확인

```bash
source ros2_ws/install/setup.bash
ros2 node list
```

**출력 예시 (연결됨):**
```
/dsr01/dsr_controller
/dsr01/state_publisher
/camera/camera
/rviz
```

### 2️⃣ 로봇 토픽 확인

```bash
ros2 topic list | grep dsr
```

**출력 예시:**
```
/dsr01/joint_states
/dsr01/moveit_controller/follow_joint_trajectory/cancel
/dsr01/moveit_controller/follow_joint_trajectory/feedback
...
```

### 3️⃣ 카메라 토픽 확인

```bash
ros2 topic list | grep camera
```

**출력 예시:**
```
/camera/camera/aligned_depth_to_color/image_raw
/camera/camera/color/image_raw
/camera/camera/color/camera_info
...
```

---

## 🔧 트러블슈팅

### 문제 1: "Host is not reachable"

**원인:** 로봇 IP 주소가 잘못되었거나 네트워크에 연결되지 않음

**해결:**
```bash
# 1. 로봇 IP 확인
ping 192.168.1.100

# 2. 네트워크 인터페이스 확인
ip addr show

# 3. 로봇 연결 확인
nmap -p 12345 192.168.1.100
```

**수정:**
```bash
./connect_real_robot.sh <올바른_IP> 12345 m0609
```

---

### 문제 2: "Port already in use"

**원인:** 포트가 이미 다른 프로세스에서 사용 중

**해결:**
```bash
# 1. 포트 점유 프로세스 확인
lsof -i :12345

# 2. 해당 프로세스 종료
kill -9 <PID>

# 3. 다시 연결
./connect_real_robot.sh 192.168.1.100 12345 m0609
```

---

### 문제 3: "Module not found: dsr_bringup2"

**원인:** ROS2 패키지가 빌드되지 않음

**해결:**
```bash
cd /home/rokey/DUM-E/ros2_ws
colcon build
source install/setup.bash
```

---

### 문제 4: "Cannot connect to robot"

**원인:** 로봇이 실제로 꺼져있거나 응답하지 않음

**확인:**
```bash
# 1. 로봇 전원 확인 (물리적으로 확인)
# 2. 네트워크 케이블 연결 확인
# 3. 로봇 고정 IP 설정 확인

# 4. 로봇 포트 스캔
nmap -p 1- 65535 192.168.1.100

# 5. 로봇 서비스 상태 (만약 제어기가 있다면)
systemctl status doosan-robot
```

---

## 📊 연결 후 확인 사항

### ✅ 로봇이 제대로 연결되면:

1. **RViz 창이 열림** - 로봇 모델 시각화
2. **/dsr01 토픽들이 생성됨** - `ros2 topic list`에서 확인
3. **콘솔에 "Robot connected" 메시지**

### 로봇 상태 모니터링

```bash
# 1. 로봇 관절 상태
ros2 topic echo /dsr01/joint_states

# 2. 로봇 TCP 포즈
ros2 service call /dsr01/get_current_posx std_srvs/srv/Empty

# 3. 그리퍼 상태
ros2 topic echo /dsr01/gripper/state
```

---

## 🔄 전체 연결 프로세스

```
┌─────────────────────────────────────────────────────┐
│ 1. ROS2 환경 설정                                    │
│    source ros2_ws/install/setup.bash                │
└──────────────────┬──────────────────────────────────┘
                   ↓
┌─────────────────────────────────────────────────────┐
│ 2. 로봇 연결 시작                                    │
│    ./connect_real_robot.sh                          │
└──────────────────┬──────────────────────────────────┘
                   ↓
┌─────────────────────────────────────────────────────┐
│ 3. RViz 실행 (자동)                                 │
│    - 로봇 모델 표시                                 │
│    - 카메라 스트림 (선택)                          │
└──────────────────┬──────────────────────────────────┘
                   ↓
┌─────────────────────────────────────────────────────┐
│ 4. YOLO 프로그램 준비                               │
│    다른 터미널에서 python3 yolo_pickplace.py 실행  │
└─────────────────────────────────────────────────────┘
```

---

## 📝 로봇 연결 체크리스트

연결 전 확인:
- [ ] 로봇이 전원에 연결되어 있는가?
- [ ] 로봇이 네트워크에 연결되어 있는가?
- [ ] 로봇 IP 주소를 알고 있는가? (기본값: 192.168.1.100)
- [ ] ROS2가 설치되어 있는가?
- [ ] `ros2_ws/install/setup.bash`가 존재하는가?
- [ ] RealSense 카메라가 연결되어 있는가?

연결 후 확인:
- [ ] RViz 창이 열렸는가?
- [ ] `ros2 node list`에서 dsr 노드가 보이는가?
- [ ] `ros2 topic list`에서 /dsr01 토픽이 보이는가?
- [ ] 로봇이 움직일 준비가 되어 있는가?

---

## 🆘 추가 도움말

### 로봇 IP 찾기

```bash
# 네트워크 스캔
nmap -p 12345 192.168.1.0/24

# 또는 로봇 제어기의 웹 인터페이스 접속
# http://192.168.1.100 (기본값)
```

### ROS2 네트워크 문제 해결

```bash
# 1. ROS_DOMAIN_ID 설정 (같은 네트워크에서 실행)
export ROS_DOMAIN_ID=0

# 2. 방화벽 확인
sudo ufw status
sudo ufw allow 4700:4800/udp  # ROS2 포트

# 3. 멀티캐스트 확인
ping 224.0.0.251
```

---

**마지막 업데이트:** 2025-12-09
