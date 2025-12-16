# 🎯 RViz에서 YOLO 마커 보이기

## ✅ 문제 해결됨

**문제:** Fixed Frame이 `base_link`여서 로봇 미연결 시 frame이 없음  
**해결:** Frame을 `camera_color_optical_frame`으로 변경

---

## 🚀 마커를 보는 방법 (3단계)

### **Step 1: YOLO 프로그램 실행**

**터미널 1:**
```bash
cd /home/rokey/DUM-E
source ros2_ws/install/setup.bash
python3 yolo_pickplace.py
```

### **Step 2: RViz 실행**

**터미널 2:**
```bash
cd /home/rokey/DUM-E
rviz2 -d rviz_yolo_config.rviz
```

### **Step 3: 마커 확인**

RViz 화면에 **초록/노랑/빨강 구**가 나타나야 합니다!

---

## 🎨 마커 설명

| 색상 | 신뢰도 | 의미 |
|---|---|---|
| 🟢 **초록** | > 0.8 | 매우 높음 |
| 🟡 **노랑** | 0.6 ~ 0.8 | 중간 |
| 🔴 **빨강** | < 0.6 | 낮음 |

---

## 🔧 만약 마커가 안 보이면?

### **체크 1: RViz Fixed Frame 확인**
1. RViz 왼쪽 `Global Options` 클릭
2. `Fixed Frame` 확인 → **`camera_color_optical_frame`** 이어야 함
3. 다르면 드롭다운에서 선택

### **체크 2: Marker Array 활성화 확인**
1. `Displays` 패널에서 `Marker Array` 찾기
2. ✅ 체크박스 활성화
3. Topic: `/visualization_marker_array` 확인

### **체크 3: 마커 데이터 실제 발행 확인**
```bash
# 다른 터미널에서
source /home/rokey/DUM-E/ros2_ws/install/setup.bash
ros2 topic echo /visualization_marker_array
```

데이터가 나오면 발행 중!

### **체크 4: 카메라 frame 존재 확인**
```bash
source /home/rokey/DUM-E/ros2_ws/install/setup.bash
ros2 topic list | grep tf
```

`/tf` 또는 `/tf_static` 있어야 함

---

## 🎯 카메라 프레임이 없으면?

카메라 노드를 다시 시작하세요:

```bash
# 터미널 3에서
ros2 launch realsense2_camera rs_align_depth_launch.py \
  depth_module.depth_profile:=640x480x30 \
  rgb_camera.color_profile:=640x480x30 \
  initial_reset:=true \
  align_depth.enable:=true
```

---

## 💡 팁

### 마커 크기 조정
`yolo_pickplace.py`에서:
```python
sphere.scale.x = 0.1  # 0.05에서 0.1로 변경 (더 크게)
```

### 마커 색상 변경
```python
if conf > 0.8:
    sphere.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0)  # 초록
```

### 마커 텍스트 표시
RViz Marker Array 디스플레이에서 텍스트도 같이 표시됩니다:
- 물체 이름
- 신뢰도 (conf: 0.82)

---

## 📊 전체 구조

```
OpenCV 윈도우          RViz 창
┌──────────────┐      ┌────────────────┐
│  카메라 영상  │      │  3D 공간       │
│ ┌──────────┐ │      │ ┌──────────┐  │
│ │scissors  │ │  ────→ │  🟢 구   │  │
│ │ conf:0.8 │ │      │ │텍스트   │  │
│ └──────────┘ │      │ └──────────┘  │
└──────────────┘      └────────────────┘
  (2D 이미지)           (3D 좌표)
```

---

## ✨ 성공했을 때 화면

```
RViz 좌측 패널:
✓ Global Options
  ✓ Fixed Frame: camera_color_optical_frame
✓ Grid
✓ MarkerArray
  Topic: /visualization_marker_array
```

중앙 화면:
- 카메라 좌표계 원점 (원)
- 감지된 물체 마커 (초록/노랑/빨강 구)
- 물체 이름 텍스트

---

**이제 RViz를 다시 열어서 마커를 확인하세요!** 🚀

```bash
rviz2 -d /home/rokey/DUM-E/rviz_yolo_config.rviz
```
