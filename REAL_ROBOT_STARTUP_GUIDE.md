# Real Robot YOLO Pick & Place 실행 가이드

## 📋 개요

이 문서는 실제 로봇(real-robot)과 연결한 후 YOLO 객체 감지 및 pick & place를 실행하는 방법을 설명합니다.

---

## 🚀 빠른 시작 (추천)

### 방법 1: 통합 스크립트 (가장 간단)

```bash
cd /home/rokey/DUM-E
./start_real_robot_yolo_full.sh
```

**기능:**
- ✅ Real Robot 자동 연결
- ✅ ROS2 환경 자동 설정
- ✅ YOLO PickPlace 자동 실행
- ✅ (선택) RViz 자동 실행

---

## 📦 단계별 실행 방법

### 방법 2: 스크립트 사용 (기본)

```bash
cd /home/rokey/DUM-E
./start_real_robot_yolo.sh
```

**Step 1:** Real Robot 연결  
**Step 2:** ROS2 환경 설정  
**Step 3:** YOLO PickPlace 실행  

---

### 방법 3: 수동 실행 (세밀한 제어 원할 때)

#### 터미널 1: Real Robot 연결
```bash
real-robot
```

#### 터미널 2: ROS2 환경 설정 및 YOLO 실행
```bash
cd /home/rokey/DUM-E
source ros2_ws/install/setup.bash
python3 yolo_pickplace.py
```

#### 터미널 3: (선택사항) RViz 실시간 시각화
```bash
cd /home/rokey/DUM-E
rviz2 -d rviz_yolo_config.rviz
```

---

## ⚙️ 프로그램 실행 중 조작

### YOLO 윈도우 조작

| 키 | 기능 |
|---|---|
| **p** | 감지된 물체를 집어서 드롭 (pick & drop) |
| **ESC** | 프로그램 종료 |

### RViz 화면

- **좌측 마우스 드래그**: 카메라 회전
- **마우스 휠**: 줌 인/아웃
- **중간 마우스 드래그**: 팬
- **우측 마우스**: 메뉴

---

## 🔍 시각화 정보

### RViz 마커

감지된 물체는 다음과 같이 표시됩니다:

| 색상 | 의미 | 신뢰도 |
|---|---|---|
| 🟢 **초록** | 높은 신뢰도 | > 0.8 |
| 🟡 **노랑** | 중간 신뢰도 | 0.6 ~ 0.8 |
| 🔴 **빨강** | 낮은 신뢰도 | < 0.6 |

### 콘솔 로그

```
📡 [1/3] Real Robot에 연결 중...
✅ Real Robot 연결 완료!

🔧 [2/3] ROS2 환경 설정 중...
✅ ROS2 환경 설정 완료!

🚀 [3/3] YOLO PickPlace 프로그램 시작...
[YOLO] Target: cup, conf=0.85, pixel=(320,240)
🎯 RViz 마커 발행: cup, conf=0.85, pos=[...,..,...]
[MOVE] Pick&Place → base(...)
```

---

## 🛠️ 트러블슈팅

### Q1: `real-robot` 명령어를 찾을 수 없음

**원인:** real-robot이 PATH에 등록되지 않음

**해결:**
```bash
# 1. 현재 로그인 셸 확인
echo $SHELL

# 2. 셸 설정 파일 수정 (~/.bashrc 또는 ~/.zshrc)
nano ~/.bashrc

# 3. 다음 줄 추가 (real-robot의 경로 확인 필요)
export PATH="/path/to/real-robot:$PATH"

# 4. 셸 재로드
source ~/.bashrc
```

---

### Q2: ROS2 환경이 설정되지 않음

**원인:** ros2_ws를 빌드하지 않음

**해결:**
```bash
cd /home/rokey/DUM-E/ros2_ws
colcon build
source install/setup.bash
```

---

### Q3: YOLO 모델을 찾을 수 없음

**오류 메시지:**
```
FileNotFoundError: [Errno 2] No such file or directory: '/home/ilhoon/...'
```

**해결:** `yolo_pickplace.py`에서 MODEL_PATH를 수정
```python
MODEL_PATH = "/path/to/yolov8s-worldv2.pt"
```

---

### Q4: 카메라 intrinsics를 받지 못함

**원인:** 카메라 노드가 실행되지 않음

**해결:**
```bash
# ROS2 토픽 확인
ros2 topic list | grep camera

# 카메라 노드 직접 실행
ros2 launch realsense2_camera rs_launch.py
```

---

### Q5: pick & drop 실행 후 로봇이 움직이지 않음

**가능한 원인:**
1. 로봇이 실제로 연결되지 않음
2. 변환 행렬 (T_gripper2camera.npy) 오류
3. 로봇 IP/포트 설정 오류

**확인:**
```bash
# 로봇 상태 확인
ros2 topic list | grep dsr
ros2 service list | grep dsr
```

---

## 📊 스크립트 상태 확인

### 프로세스 모니터링

```bash
# 현재 실행 중인 Python 프로세스 확인
ps aux | grep python3

# YOLO 프로세스 찾기
ps aux | grep yolo_pickplace

# RViz 프로세스 찾기
ps aux | grep rviz2
```

### 프로세스 강제 종료

```bash
# PID로 종료
kill -9 <PID>

# 프로세스 이름으로 종료
pkill -f yolo_pickplace
pkill -f rviz2
```

---

## 🔧 고급 설정

### 환경 변수 설정

```bash
# ROS2 Domain ID 설정 (네트워크 격리)
export ROS_DOMAIN_ID=0

# 로그 레벨 조정 (DEBUG, INFO, WARN, ERROR)
export RCL_LOG_LEVEL=INFO
```

### 스크립트 커스터마이징

#### `start_real_robot_yolo.sh` 수정

```bash
# YOLO 설정 변경
nano start_real_robot_yolo.sh

# MODEL_PATH 수정
# ROBOT_ID 수정
# 속도/가속도 설정 변경
```

---

## 📝 완전 자동화 (tmux 사용)

여러 터미널을 자동으로 관리하고 싶다면:

```bash
#!/bin/bash
# auto_startup.sh

# tmux 세션 생성
tmux new-session -d -s robot_control

# 각 창에서 명령어 실행
tmux send-keys -t robot_control "real-robot" Enter
sleep 5
tmux new-window -t robot_control
tmux send-keys -t robot_control "cd /home/rokey/DUM-E && source ros2_ws/install/setup.bash && python3 yolo_pickplace.py" Enter
sleep 3
tmux new-window -t robot_control
tmux send-keys -t robot_control "cd /home/rokey/DUM-E && rviz2 -d rviz_yolo_config.rviz" Enter

# 모든 창 보기
tmux attach-session -t robot_control
```

실행:
```bash
chmod +x auto_startup.sh
./auto_startup.sh
```

---

## 📞 도움말

### 자주 사용되는 명령어

```bash
# ROS2 노드 목록
ros2 node list

# ROS2 토픽 목록
ros2 topic list

# 특정 토픽 모니터링
ros2 topic echo /visualization_marker_array

# 로봇 관절 상태
ros2 topic echo /dsr01/joint_states

# 카메라 정보
ros2 service call /camera/get_camera_intrinsics std_srvs/srv/Empty {}
```

---

## ✅ 체크리스트

프로그램 시작 전 확인:

- [ ] Real Robot이 네트워크에 연결되어 있는가?
- [ ] RealSense 카메라가 USB에 연결되어 있는가?
- [ ] `ros2_ws/install/setup.bash`가 존재하는가?
- [ ] `yolo_pickplace.py`의 MODEL_PATH가 올바른가?
- [ ] `T_gripper2camera.npy` 파일이 존재하는가?
- [ ] 로봇 IP/포트 설정이 올바른가?

---

## 📚 관련 파일

- 메인 프로그램: `/home/rokey/DUM-E/yolo_pickplace.py`
- RViz 설정: `/home/rokey/DUM-E/rviz_yolo_config.rviz`
- 시작 스크립트 (기본): `/home/rokey/DUM-E/start_real_robot_yolo.sh`
- 시작 스크립트 (통합): `/home/rokey/DUM-E/start_real_robot_yolo_full.sh`
- 변환 행렬: `/home/rokey/DUM-E/ros2_ws/src/perception/config/T_gripper2camera.npy`

---

**마지막 업데이트:** 2025-12-09
