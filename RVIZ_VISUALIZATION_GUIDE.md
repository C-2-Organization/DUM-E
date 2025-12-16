# RViz에서 YOLO 감지 객체 3D 좌표 시각화 가이드

## 📋 수정 사항

### 1. **의존성 추가** (imports)
- `visualization_msgs.msg` - RViz 마커 시각화
- `geometry_msgs.msg` - 3D 포인트/좌표
- `std_msgs.msg` - 색상 정보

### 2. **TestNode 초기화 추가**
```python
self.marker_pub = self.create_publisher(MarkerArray, "/visualization_marker_array", 10)
self.marker_id = 0
```

### 3. **publish_marker() 메서드 추가**
- 감지된 객체를 RViz에 3D 구(Sphere)로 표시
- 신뢰도(confidence)에 따라 색상 변경:
  - 🟢 **Green** (conf > 0.8)
  - 🟡 **Yellow** (0.6 < conf ≤ 0.8)  
  - 🔴 **Red** (conf ≤ 0.6)
- 텍스트 마커로 객체 이름 및 신뢰도 표시

### 4. **메인 루프(run) 수정**
- 타겟 감지 시마다 RViz 마커 퍼블리시
- 카메라 좌표 → 베이스 좌표로 변환 후 표시

---

## 🚀 실행 방법

### **1단계: RViz 실행**
```bash
cd /home/rokey/DUM-E
rviz2 -d rviz_yolo_config.rviz
```

### **2단계: 로봇/카메라 런치 (다른 터미널)**
```bash
source ros2_ws/install/setup.bash
# 로봇 시뮬레이션 또는 실제 로봇 시작
```

### **3단계: YOLO PickPlace 실행 (또 다른 터미널)**
```bash
cd /home/rokey/DUM-E
python yolo_pickplace.py
```

---

## 🎯 RViz 마커 설명

### **표시되는 마커:**
1. **구(Sphere)** - 감지된 물체의 3D 위치
   - 크기: 5cm 반지름
   - 색상: 신뢰도에 따라 동적 변경

2. **텍스트** - 물체 이름과 신뢰도
   - 구 위에 0.1m 위치에 표시
   - 예: `cup\n(conf: 0.85)`

---

## 📊 좌표 변환 과정

```
카메라 좌표 (픽셀 + 깊이)
    ↓ (내부 파라미터 사용)
카메라 좌표 (3D mm)
    ↓ (gripper2cam 변환)
그리퍼 좌표
    ↓ (base2gripper 변환)
로봇 베이스 좌표 ✅ (RViz에서 표시)
```

---

## ⚙️ 커스터마이징

### **마커 크기 조정**
`publish_marker()` 메서드에서:
```python
sphere.scale.x = 0.05  # 변경 (단위: m)
```

### **색상 변경**
```python
if conf > 0.8:
    sphere.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0)  # RGB 0~1 범위
```

### **마커 수명 설정** (선택사항)
```python
sphere.lifetime = rclpy.duration.Duration(seconds=1.0)  # 1초 후 자동 제거
```

---

## ✅ 트러블슈팅

### **Q: RViz에 마커가 보이지 않음**
- ✓ Fixed Frame을 `base_link`로 설정했는가?
- ✓ MarkerArray 토픽 확인: `/visualization_marker_array`
- ✓ YOLO 감지 성공 여부 확인 (콘솔 로그 확인)

### **Q: 마커 위치가 이상함**
- ✓ `T_gripper2camera.npy` 파일 확인
- ✓ 카메라 intrinsics 정확성 확인
- ✓ 좌표 변환 로직 재검토

### **Q: 마커가 많이 쌓임**
- 현재: 계속 누적됨
- 해결: `lifetime` 속성 추가 또는 마커 ID 관리 개선

---

## 📝 추가 기능 (옵션)

마커를 발행하지 않고 싶으면 `run()` 메서드에서 이 부분을 주석 처리:
```python
# self.publish_marker(base_pos, conf, best_name)
```
