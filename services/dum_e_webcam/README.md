# Dum-E Webcam Services Guide

Dum-E 프로젝트의 `webcam/services`는 고정 웹캠(CCTV)을 기반으로
시각 인식 → 좌표 해석 → 상황 판단 → 로봇 행동 디스패치를 담당하는 핵심 모듈입니다.

본 문서는 전체 폴더 구조, 각 파일의 역할, 사용 흐름, import 방식까지
개발자가 바로 사용할 수 있도록 정리한 가이드입니다.

---

## 1. 전체 폴더 구조

webcam/services/
├── __init__.py
│
├── io/
│   └── camera_capture.py
│
├── vision/
│   ├── motion_detector.py
│   └── yolo_locator.py
│
├── table/
│   ├── table_roi.py
│   ├── hole_detector.py
│   └── hole_grid_warp.py
│
├── geometry/
│   └── homography.py
│
├── mapping/
│   └── location_mapper.py
│
├── llm/
│   └── gpt_situation.py
│
└── actions/
    └── action_dispatcher.py

---

## 2. 전체 처리 흐름 개요

CameraCapture 로 웹캠 프레임을 수집한 뒤
MotionDetector 또는 YOLO 기반 인식을 통해 이상 상황을 감지합니다.

감지된 좌표는
테이블 ROI 판별 → 홀 그리드 워핑 → 셀 위치 계산 →
로봇 좌표 변환 과정을 거칩니다.

이후 프레임은 GPT Vision 모델로 전달되어
상황 설명 및 추천 행동(JSON)을 생성하고,
Action Dispatcher가 이를 실제 로봇 동작으로 연결합니다.

---

## 3. 모듈별 상세 설명

### 3.1 io / camera_capture.py
- OpenCV 기반 웹캠 프레임 입력
- 프레임 읽기 실패 시 안전하게 None 반환

사용 예:
CameraCapture(device=0, width=1280, height=720)

---

### 3.2 vision / motion_detector.py
- 프레임 차이를 이용한 움직임 감지
- bbox, 중심 좌표 반환
- CCTV 감시용 이상 감지에 사용

---

### 3.3 vision / yolo_locator.py
- YOLOWorld 모델 기반 객체 탐지
- 객체 중심 좌표 계산
- hand, cup, bottle 등 객체 위치 인식에 사용

---

### 3.4 table / table_roi.py
- table_roi.json 기반 테이블 영역 정의
- 특정 좌표가 테이블 내부인지 판별

주요 기능:
point_in_table(x, y)

---

### 3.5 table / hole_detector.py
- 기준 홀 좌표(JSON) 로딩
- 테이블 기준점 관리
- 디버그용 시각화 가능

---

### 3.6 table / hole_grid_warp.py
- 테이블 ROI를 정사각형 좌표계로 워핑
- 입력 좌표를 가장 가까운 홀 셀로 매핑

주요 기능:
locate_point_to_cell_warped(px, py)

---

### 3.7 geometry / homography.py
- 4점 기반 Homography 계산
- 픽셀 좌표 변환

주요 기능:
compute_homography()
warp_point()

---

### 3.8 mapping / location_mapper.py
- 테이블 셀(홀 4개) 기준으로
  로봇 좌표 평균값 계산

주요 기능:
cell_to_robot_xy(cell)

---

### 3.9 llm / gpt_situation.py
- 프레임을 base64로 변환
- OpenAI Vision 모델에 상황 질의
- JSON 형태로 상황 설명 및 추천 행동 반환

반환 예:
{
  "scene_summary": "Water is spilled on the table",
  "risk_level": "medium",
  "recommended_action": "clean_spill"
}

---

### 3.10 actions / action_dispatcher.py
- GPT 결과(JSON)를 해석
- ROS2 서비스 또는 토픽으로 로봇 행동 디스패치
- 실제 로봇 제어 진입 지점

---

## 4. 통합 Import 방식

services/__init__.py에서 주요 기능을 재노출하고 있어
외부에서는 다음과 같이 간단히 사용 가능합니다.

from webcam.services import
    CameraCapture,
    MotionDetector,
    locate_point_to_cell_warped,
    cell_to_robot_xy,
    analyze_situation

---

## 5. Worker / Node 사용 예시 흐름

1. CameraCapture로 프레임 획득
2. MotionDetector 또는 YOLO로 이상 감지
3. 좌표 → 테이블 ROI → 홀 셀 변환
4. 로봇 좌표 계산
5. GPT로 상황 판단
6. Action Dispatcher로 로봇 행동 수행

---

## 6. 설계 원칙

- Vision / Geometry / LLM / Action 완전 분리
- GPT는 판단만 수행, 로봇 제어는 절대 하지 않음
- 모든 실제 행동은 Action Dispatcher에서만 수행
- 각 모듈은 단일 책임 원칙을 유지

---

## 7. 향후 확장 방향

- hole grid 워핑 캐싱 최적화
- 상황 우선순위 큐 도입
- 위험 상황 전용 GPT 프롬프트 분리
- 멀티 카메라 지원

---

Maintained by Dum-E Project
