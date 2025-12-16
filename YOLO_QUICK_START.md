# 🎯 YOLO 물체 감지 사용법 (즉시 시작 가능!)

## ✅ 해결된 문제

| 문제 | 원인 | 해결 |
|---|---|---|
| `ModuleNotFoundError: onrobot` | ROS2 환경 미설정 | sys.path에 utils 경로 추가 |
| `FileNotFoundError: T_gripper2camera.npy` | 파일 경로 오류 | 자동으로 여러 경로에서 검색 |
| `MODEL_PATH 오류` | 없는 디렉토리 참조 | 자동 다운로드 모드로 변경 |
| 로봇 연결 오류 | 로봇 미연결 | 로봇 없어도 카메라로 작동하도록 수정 |
| 그리퍼 오류 | 라이브러리 미설치 | 그리퍼 없어도 작동하도록 수정 |

---

## 🚀 지금 바로 시작하기 (가장 간단!)

```bash
cd /home/rokey/DUM-E
./run_yolo_simple.sh
```

**이것만으로 시작됩니다:**
- ✅ 카메라 영상 캡처
- ✅ YOLO로 물체 감지
- ✅ OpenCV 윈도우에 결과 표시
- ✅ RViz에 3D 마커 발행

---

## 📊 프로그램 시작 시 보이는 것

### 1️⃣ **OpenCV 윈도우** - 실시간 카메라 + 감지 결과

```
┌─────────────────────────────────┐
│  카메라 영상                     │
│  ┌─────────────────────────┐   │
│  │  [scissors]  conf: 0.52 │   │  ← 바운딩 박스
│  │         ●               │   │  ← 중심점
│  └─────────────────────────┘   │
└─────────────────────────────────┘
```

**표시 정보:**
- 🟩 **바운딩 박스** - 감지된 물체 주변
- 🎯 **중심점** - 초록 점
- 📝 **클래스 이름** - "scissors", "cup" 등
- 📊 **신뢰도** - 0.52, 0.85 등

### 2️⃣ **콘솔 로그** - 감지 정보 출력

```
[YOLO] Target: scissors, conf=0.52, pixel=(415,275)
[YOLO] Target: cup, conf=0.85, pixel=(320,240)
🎯 RViz 마커 발행: scissors, conf=0.52, pos=[383.2, 32.9, -165.7]
```

### 3️⃣ **RViz (선택)** - 3D 시각화

```bash
# 다른 터미널에서 실행
rviz2 -d /home/rokey/DUM-E/rviz_yolo_config.rviz
```

**마커:**
- 🟢 **초록 구** - 신뢰도 높음 (>0.8)
- 🟡 **노랑 구** - 신뢰도 중간 (0.6-0.8)
- 🔴 **빨강 구** - 신뢰도 낮음 (<0.6)

---

## 🎮 프로그램 조작

| 키 | 기능 | 현재 상태 |
|---|---|---|
| **p** | Pick & Drop 실행 | ⚠️ 로봇 연결 필요 |
| **ESC** | 프로그램 종료 | ✅ 작동 |

---

## 📋 감지되는 물체 종류

현재 설정된 클래스:

```python
YOLO_CLASSES = [
    "person",       # 사람
    "cup",          # 컵
    "scissors",     # 가위
    "box cutter",   # 커터
    "bottle",       # 병
    "laptop"        # 노트북
]

# Pick & Place 대상
TARGET_CLASSES = {"cup", "scissors", "box cutter"}
```

---

## 🔄 다음 단계 (로봇과 함께 사용)

### Step 1: 로봇 연결

```bash
./connect_real_robot.sh
```

또는:
```bash
ros2 launch dsr_bringup2 dsr_bringup2_rviz.launch.py \
    mode:=real \
    host:=192.168.1.100 \
    port:=12345 \
    model:=m0609
```

### Step 2: 그리퍼 의존성 설치 (선택)

```bash
pip install pymodbus
```

### Step 3: YOLO 실행

```bash
./run_yolo_simple.sh
```

이제 'p' 키로 pick & drop이 작동합니다!

---

## 🔍 감지가 안 되면?

### 체크리스트

- [ ] 카메라가 연결되어 있는가?
  ```bash
  ros2 topic list | grep camera
  ```

- [ ] 감지하려는 물체가 화면에 보이는가?
  ```bash
  # OpenCV 윈도우 확인
  ```

- [ ] 신뢰도(confidence)가 임계값보다 높은가?
  ```python
  YOLO_CONF_TH = 0.5  # 50% 이상의 신뢰도
  ```

### 설정 변경 (필요시)

```bash
nano yolo_pickplace.py

# 수정 가능한 부분:
YOLO_CLASSES = [...]        # 감지 클래스 추가
YOLO_CONF_TH = 0.5          # 감지 임계값 조정
TARGET_CLASSES = {...}      # pick & place 대상 변경
```

---

## 💡 팁

### 1. RViz와 함께 실행

**터미널 1:**
```bash
cd /home/rokey/DUM-E
./run_yolo_simple.sh
```

**터미널 2:**
```bash
rviz2 -d /home/rokey/DUM-E/rviz_yolo_config.rviz
```

### 2. 감지 결과 필터링

신뢰도가 낮은 감지를 무시하려면:
```python
YOLO_CONF_TH = 0.7  # 70% 이상만 감지
```

### 3. 로깅 레벨 조정

```bash
export RCL_LOG_LEVEL=DEBUG  # 상세한 로그
export RCL_LOG_LEVEL=WARN   # 경고만 표시
```

---

## 📂 주요 파일

| 파일 | 설명 |
|---|---|
| `run_yolo_simple.sh` | 🎯 가장 간단한 시작 스크립트 |
| `yolo_pickplace.py` | 메인 YOLO 프로그램 |
| `rviz_yolo_config.rviz` | RViz 설정 파일 |
| `connect_real_robot.sh` | 로봇 연결 스크립트 |
| `diagnose_robot.sh` | 연결 상태 진단 스크립트 |

---

## 🆘 문제 해결

### 문제: "Camera not found"

```bash
# 카메라 노드 확인
ros2 launch realsense2_camera rs_align_depth_launch.py
```

### 문제: "YOLO 모델 다운로드 느림"

첫 실행만 시간이 걸림 (약 30초). 이후는 캐시에서 로드되어 빠름.

### 문제: "메모리 부족"

```bash
# 더 작은 모델 사용
MODEL_PATH = "yolov8n-worldv2.pt"  # nano 버전 (더 빠름)
```

---

## 🎉 성공 표시

프로그램이 정상 작동하면:

```
📷 camera intrinsics 수신 완료: {...}
✅ 변환 행렬 로드됨: ...
Loading weights...
▶ YOLO_PickPlace 실행 중... 'p' 누르면 픽앤플레이스, ESC 누르면 종료
[YOLO] Target: scissors, conf=0.52, pixel=(415,275)
🎯 RViz 마커 발행: scissors, conf=0.52, pos=[...]
```

---

## 📝 설정 커스터마이징

### 새로운 물체 감지하기

```python
# yolo_pickplace.py 수정
YOLO_CLASSES = ["person", "cup", "scissors", "box cutter", "bottle", "laptop", "pen"]
TARGET_CLASSES = {"cup", "scissors", "box cutter", "pen"}  # pen 추가
```

### 감지 정확도 조정

```python
YOLO_CONF_TH = 0.7   # 더 엄격하게 (70% 이상만)
YOLO_CONF_TH = 0.3   # 더 관대하게 (30% 이상도 포함)
```

### 마커 색상 변경

```python
# publish_marker() 메서드에서
if conf > 0.8:
    sphere.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0)  # Green
```

---

**이제 바로 실행하세요!** 🚀
```bash
cd /home/rokey/DUM-E
./run_yolo_simple.sh
```
