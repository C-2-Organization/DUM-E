# DUM-E 자동 복구 시스템 (Auto Recovery System)

## 개요

DUM-E의 자동 복구 시스템은 로봇이 SAFE_STOP 상태에 진입했을 때 자동으로 복구 시퀀스를 수행하고, 드라이버 무응답 시 자동 재시작을 처리하는 기능입니다.

## 주요 기능

### 1. 자동 SAFE_STOP 복구
- **충돌 감지**: 로봇이 SAFE_STOP 상태로 진입하면 자동으로 감지
- **복구 시퀀스**: 6단계 복구 프로세스 자동 실행
  1. SAFE_STOP 리셋
  2. RECOVERY 모드 진입
  3. Z축 상승 (바닥 충돌 시)
  4. RECOVERY 완료 처리
  5. RECOVERY 모드 해제
  6. 서보 ON
- **상태 확인**: STANDBY 상태 도달까지 자동 대기 및 검증

### 2. 드라이버 자동 재시작
- **무응답 감지**: 로봇 드라이버가 일정 시간 응답하지 않으면 자동 감지 (기본: 1.6초)
- **자동 재기동**: 전체 통합 브링업 시스템 재시작
- **좀비 프로세스 정리**: RViz, ros2_control, robot_state_publisher 등 강제 종료
- **FastDDS 캐시 정리**: 공유 메모리 충돌 방지
- **서비스 복구 대기**: 재시작 후 서비스 준비 확인

## 아키텍처

```
┌─────────────────────────────────────────────────────────┐
│                   통합 브링업 런치                         │
│          (dum_e_bringup.launch.py)                       │
│                                                           │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌─────────┐ │
│  │ DSR      │  │ RealSense│  │ RViz     │  │ Motion  │ │
│  │ Bringup  │  │ Camera   │  │          │  │ / Skill │ │
│  └──────────┘  └──────────┘  └──────────┘  └─────────┘ │
└─────────────────────────────────────────────────────────┘
                            ▲
                            │ 모니터링 + 재시작
                            │
┌───────────────────────────┴─────────────────────────────┐
│             복구 노드 (독립 실행)                          │
│          (robot_care.launch.py)                          │
│                                                           │
│  ┌────────────────────────────────────────────────┐     │
│  │  robot_state_care_node                         │     │
│  │  • 상태 폴링 (0.2초 간격)                       │     │
│  │  • SAFE_STOP 감지 → auto_recover() 호출        │     │
│  │  • 드라이버 무응답 감지 → 재시작                │     │
│  └────────────────────────────────────────────────┘     │
│                      ▼                                    │
│  ┌────────────────────────────────────────────────┐     │
│  │  CollisionRecovery (dum_e_motion)              │     │
│  │  • 6단계 복구 시퀀스 실행                       │     │
│  │  • 그리퍼 상태별 분기 처리                      │     │
│  │  • 상태 전환 검증                               │     │
│  └────────────────────────────────────────────────┘     │
└───────────────────────────────────────────────────────┘
```

## 사용 방법

### 기본 실행

```bash
# 터미널 1: 복구 노드 실행 (독립)
cd ~/DUM-E/ros2_ws
source install/setup.bash
ros2 launch dum_e_bringup robot_care.launch.py

# 터미널 2: 통합 브링업 실행
source ~/DUM-E/ros2_ws/install/setup.bash
ros2 launch dum_e_bringup dum_e_bringup.launch.py with_care:=false
```

### 파라미터 튜닝

```bash
# 응답 속도 조정
ros2 launch dum_e_bringup robot_care.launch.py \
  monitor_interval:=0.2 \
  failure_threshold:=8 \
  initial_safe_stop_grace:=1.5
```

**파라미터 설명:**
- `monitor_interval`: 상태 폴링 주기 (초, 기본: 0.2)
- `failure_threshold`: 드라이버 재시작 임계값 (연속 실패 횟수, 기본: 8)
- `initial_safe_stop_grace`: 부팅 직후 SAFE_STOP 복구 유예 시간 (초, 기본: 1.5)

## 복구 시퀀스 타이밍

### 현재 설정 (최적화)

| 단계 | 동작 | 대기 시간 |
|------|------|-----------|
| 1 | SAFE_STOP 리셋 (재시도 3회) | 0.2s × 3 |
| 2 | RECOVERY 모드 진입 | 0.2s |
| 3 | Z축 상승 (조건부) | 1.5s (Jog 동작) |
| 4 | RECOVERY 완료 | 0.3s |
| 5 | RECOVERY 해제 | 0.3s |
| 6 | 서보 ON | 0.7s |
| 7 | 서비스 안정화 | 1.0s |
| 8 | STANDBY 대기 | 최대 2.0s |

**예상 총 소요 시간**: 약 3~5초 (바닥 충돌 시 5~7초)

## 로그 위치

- **복구 상세 로그**: `/tmp/recovery_monitor.log`
- **재시작 로그**: `/tmp/dsr_bringup_restart.log`

## 주요 파일

```
ros2_ws/
├── src/
│   ├── dum_e_bringup/
│   │   ├── launch/
│   │   │   ├── dum_e_bringup.launch.py      # 통합 브링업
│   │   │   └── robot_care.launch.py          # 복구 노드 (독립)
│   │   └── dum_e_bringup/
│   │       └── robot_state_care_node.py      # 상태 모니터링 + 재시작
│   └── dum_e_motion/
│       └── dum_e_motion/
│           └── recovery.py                    # 복구 시퀀스 구현
```

## 테스트 방법

### 1. SAFE_STOP 복구 테스트

```bash
# 1. 시스템 실행
ros2 launch dum_e_bringup robot_care.launch.py  # 터미널 1
ros2 launch dum_e_bringup dum_e_bringup.launch.py with_care:=false  # 터미널 2

# 2. 로봇을 움직여 충돌 유도 (또는 E-STOP 버튼)
# - 교시 펜던트에서 수동 조작
# - 또는 테스트 모션 실행

# 3. 복구 노드 로그 확인
# 기대 출력:
# ⚠️  충돌 감지! SAFE_STOP → 복구 시작
# 🔧 자동 복구 시작 (Recovery.auto_recover())
# [Recovery] 단계 1: SAFE_STOP 리셋
# [Recovery] 단계 2: RECOVERY 모드 진입
# ...
# ✅ [Recovery] 복구 시퀀스 완료!
# 📊 현재 상태: STANDBY
```

### 2. 드라이버 재시작 테스트

```bash
# 1. 시스템 실행 (위와 동일)

# 2. 로봇 연결 강제 차단 (시뮬레이션)
# - 로봇 전원 차단 또는
# - 네트워크 케이블 분리

# 3. 복구 노드 로그 확인
# 기대 출력:
# 💀 DSR 드라이버 응답 없음 (8회 연속 실패)! 자동 재시작...
# 🔄 Step 1/5: 기존 프로세스 종료...
# 🔄 Step 2/5: FastDDS cleanup...
# 🔄 Step 3/5: 로봇 컨트롤러 대기...
# 🔄 Step 4/5: 통합 런치 재시작...
# 🔄 Step 5/5: 서비스 대기...
# ✅ robot-real 재시작 완료!

# 4. 재시작 로그 확인
tail -f /tmp/dsr_bringup_restart.log
```

## 문제 해결

### 복구가 시작되지 않음

**원인**: 정상 상태를 한 번도 거치지 않은 부팅 직후 SAFE_STOP
**해결**: `initial_safe_stop_grace` 시간(기본 1.5초) 경과 후 자동 복구 시작

### SAFE_OFF 상태에서 멈춤

**원인**: 서보 ON 명령이 실패했거나 로봇이 SAFE_OFF 상태로 진입
**해결**: 
```bash
# 수동 서보 ON
ros2 service call /dsr01/system/set_robot_control dsr_msgs2/srv/SetRobotControl '{robot_control: 3}'
```

### 드라이버 재시작이 너무 빈번함

**원인**: `failure_threshold`가 너무 낮음
**해결**: 
```bash
# 임계값 증가 (예: 15회 = 3초)
ros2 launch dum_e_bringup robot_care.launch.py failure_threshold:=15
```

### 복구 속도가 너무 느림

**원인**: 기본 타이밍이 보수적으로 설정됨
**해결**:
```bash
# 빠른 설정으로 실행
ros2 launch dum_e_bringup robot_care.launch.py \
  monitor_interval:=0.1 \
  failure_threshold:=5 \
  initial_safe_stop_grace:=1.0
```

## 설계 결정

### 왜 복구 노드를 독립 실행하나요?

통합 브링업 내부에서 복구 노드를 실행하면 다음 문제가 발생합니다:
- 복구 노드가 자신을 포함한 프로세스 트리를 재시작하려 할 때 좀비 프로세스 발생
- RViz 등 GUI 프로세스가 올바르게 종료되지 않음
- 재시작 실패 시 전체 시스템 불안정

독립 실행 시:
- 복구 노드는 통합 브링업을 외부에서 관리
- 프로세스 정리 및 재시작이 명확하고 안정적
- 복구 노드 자체는 항상 살아있어 지속적 모니터링 가능

### 왜 드라이버 재시작을 기본으로 비활성화했나요?

- 복구 완료 후 자동 드라이버 재시작은 SAFE_OFF 상태로 떨어지는 부작용 발견
- 대부분의 SAFE_STOP은 복구 시퀀스만으로 해결 가능
- 필요 시 `recovery.restart_driver_after_complete = True`로 수동 활성화 가능

## 개선 이력

### v1.0 (2025-12-16)
- ✅ 독립 복구 노드 패턴 적용
- ✅ 6단계 복구 시퀀스 구현
- ✅ 드라이버 자동 재시작 기능
- ✅ 상태 폴링 및 타이밍 최적화 (0.2초 주기)
- ✅ FastDDS 캐시 정리 로직
- ✅ 좀비 프로세스 방지 (pkill -f 패턴)
- ✅ 로그 가시성 향상 (flush=True)
- ✅ 초기 SAFE_STOP 복구 지원

## 참고 자료

- [Doosan Robotics ROS2 Documentation](https://github.com/doosan-robotics/doosan-robot2)
- [ROS2 Control](https://control.ros.org/)
- Original inspiration: [rokey_collabo1](https://github.com/C-2-Organization/rokey_collabo1)

## 라이선스

이 프로젝트는 Doosan Robotics의 ROS2 패키지를 기반으로 합니다.

---
**문의**: DUM-E 프로젝트 - C-2-Organization
