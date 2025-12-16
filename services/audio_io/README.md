# Jarvis Voice Assistant with HUD

Jarvis 스타일의 음성 비서 시스템 (Wakeword → STT → LLM → TTS + HUD 시각화)

## 주요 기능

### 🎯 음성 파이프라인
- **Wakeword**: Picovoice Porcupine으로 "Jarvis" 감지
- **STT**: OpenAI Whisper-1 (WebRTC VAD + 적응형 노이즈 게이팅)
- **LLM**: OpenAI GPT-4.1-mini (Jarvis 페르소나)
- **TTS**: OpenAI GPT-4o-mini-tts (Echo 보이스 + Jarvis 이펙트)

### 🎨 HUD 인터페이스
- **GIF 애니메이션**: 지속적으로 재생되는 배경 애니메이션
- **상태 표시**: STANDBY / LISTENING / PROCESSING / SPEAKING
- **실시간 오디오 피드백**: TTS 출력 시 음성 데이터 시각화

### 💬 대화 모드
- **연속 대화**: Wakeword 감지 후 자동으로 연속 대화 모드 진입
- **종료 키워드**: "그만", "끝", "됐어", "that's all" 등으로 대화 종료
- **자동 대기**: 무응답 시 자동으로 대기 모드로 복귀

## 실행 방법

### 1. 환경 설정

`.env` 파일에 필요한 설정 추가:

```bash
# OpenAI API
OPENAI_API_KEY=your_api_key_here

# Picovoice (Wakeword)
PICOVOICE_ACCESS_KEY=your_picovoice_key_here

# 마이크 설정 (선택)
MIC_DEVICE_INDEX=0  # 기본값: 0 (시스템 기본 마이크)

# STT 민감도 조정 (선택)
STT_ENERGY_THRESHOLD=100.0  # 낮을수록 민감 (기본: 100)
STT_SILENCE_SEC=1.5         # 침묵 감지 시간 (기본: 1.5초)
STT_VAD_LEVEL=1             # VAD 레벨 0-3 (기본: 1)

# TTS 설정
DUM_E_TTS_VOICE=echo        # 기본 보이스
DUM_E_TTS_EFFECT=jarvis     # Jarvis 이펙트 적용

# 언어 설정
JARVIS_LANG=ko              # ko (한국어) 또는 en (영어)

# 시작 인사 (선택)
JARVIS_GREETING=0           # 0: 비활성화 (기본), 1: 활성화
```

### 2. 실행

```bash
# 방법 1: alias 사용 (권장)
jarvis-hud

# 방법 2: 직접 실행
cd ~/rokey/DUM-E
PYTHONPATH=$PWD python3 services/audio_io/app/jarvis_ui_app.py
```

### 3. 종료

- **Ctrl+C**: 안전하게 종료
- **창 닫기**: HUD 창 닫기 버튼 클릭

## 사용법

1. **프로그램 시작**: HUD 창이 나타나고 Wakeword 대기 상태로 진입
2. **Wakeword 발화**: "Jarvis" 또는 "자비스" 말하기
3. **연속 대화**: 시스템이 "네, 말씀하세요" 응답 후 자유롭게 대화
4. **대화 종료**: "그만", "끝", "됐어", "that's all" 등 종료 키워드 말하기
5. **대기 모드**: 시스템이 대기 모드로 돌아가며 다시 Wakeword 대기

## 파일 구조

```
services/audio_io/app/
├── jarvis_ui_app.py      # 메인 애플리케이션 (통합)
├── jarvis_hud.py         # PySide6 HUD 인터페이스
├── jarvis_assistant.py   # LLM 응답 생성
├── tts_streaming.py      # TTS 스트리밍 + Jarvis 이펙트
├── stt.py                # STT (Whisper + VAD)
├── wakeword.py           # Wakeword 감지 (Porcupine)
├── mic.py                # 마이크 입력 관리
├── config.py             # 설정 관리
├── assets/
│   └── jarvis.gif        # HUD 배경 애니메이션
└── models/
    ├── jarvis.ppn        # Jarvis wakeword 모델
    └── dummy.ppn         # Dummy wakeword 모델
```

## 주요 컴포넌트

### JarvisApp (jarvis_ui_app.py)
전체 파이프라인을 통합하는 메인 애플리케이션
- Qt 이벤트 루프 관리
- Wakeword → STT → LLM → TTS 플로우 제어
- 연속 대화 모드 관리
- 종료 명령 처리

### JarvisHUD (jarvis_hud.py)
PySide6 기반 시각화 인터페이스
- GIF 애니메이션 재생 (QMovie)
- 상태 텍스트 오버레이
- 실시간 오디오 데이터 수신 (향후 확장 가능)

### StreamingTTS (tts_streaming.py)
OpenAI TTS API를 사용한 음성 합성
- 청크 단위 스트리밍 재생
- Jarvis 이펙트 (FFT EQ + bit-crush + chorus)
- 재생 중단 기능 (새 음성 시작 시 이전 음성 중단)
- HUD 콜백 (오디오 샘플 전달)

### StreamingSTT (stt.py)
OpenAI Whisper API를 사용한 음성 인식
- WebRTC VAD (Voice Activity Detection)
- 적응형 노이즈 게이팅
- 환경 변수로 민감도 조정 가능

### JarvisAssistant (jarvis_assistant.py)
LLM 기반 응답 생성
- GPT-4.1-mini 사용
- Jarvis 페르소나 (Iron Man의 AI 비서)
- 언어별 시스템 프롬프트 (한국어/영어)

## 트러블슈팅

### Jarvis가 반응하지 않을 때
1. STT 민감도 조정: `.env`에서 `STT_ENERGY_THRESHOLD` 값을 낮춤 (예: 50)
2. VAD 레벨 조정: `STT_VAD_LEVEL=0` (더 민감)
3. 마이크 확인: `MIC_DEVICE_INDEX` 값 변경

### GIF가 끊기거나 느릴 때
- GIF 파일 크기 확인 (권장: 10MB 이하)
- GIF 해상도 조정 (권장: 800x800 이하)

### 한글과 영어가 동시에 나올 때
- `.env`에서 `JARVIS_LANG` 설정 확인 (`ko` 또는 `en`)
- 시스템 프롬프트가 단일 언어로 설정되어 있는지 확인

### 종료가 제대로 안 될 때
- `Ctrl+C` 사용 (signal handler로 안전하게 종료)
- 여전히 문제가 있다면 터미널에서 강제 종료: `pkill -9 python3`

## 환경 변수 전체 목록

| 변수명 | 설명 | 기본값 | 예시 |
|--------|------|--------|------|
| `OPENAI_API_KEY` | OpenAI API 키 (필수) | - | `sk-...` |
| `PICOVOICE_ACCESS_KEY` | Picovoice API 키 (필수) | - | `your_key` |
| `MIC_DEVICE_INDEX` | 마이크 장치 인덱스 | `0` | `1`, `2` |
| `STT_ENERGY_THRESHOLD` | STT 에너지 임계값 | `100.0` | `50.0` (더 민감) |
| `STT_SILENCE_SEC` | 침묵 감지 시간 (초) | `1.5` | `1.0`, `2.0` |
| `STT_VAD_LEVEL` | VAD 레벨 (0-3) | `1` | `0` (더 민감) |
| `STT_SPEECH_RATIO_BLOCK` | 블록당 음성 비율 | `0.4` | `0.3`, `0.5` |
| `STT_SPEECH_RATIO_TOTAL` | 전체 음성 비율 | `0.2` | `0.1`, `0.3` |
| `DUM_E_TTS_VOICE` | TTS 보이스 | `echo` | `onyx`, `alloy` |
| `DUM_E_TTS_EFFECT` | TTS 이펙트 | `jarvis` | `jarvis`, `none` |
| `JARVIS_LANG` | 응답 언어 | `ko` | `en` |
| `JARVIS_GREETING` | 시작 인사 활성화 | `0` | `1` |

## 최근 업데이트

### 2024-12-16
- ✅ GIF 기반 HUD로 전환 (복잡한 그래픽 제거)
- ✅ Ctrl+C 안전 종료 구현 (signal handler)
- ✅ 종료 키워드 최적화 (자연스러운 대화 허용)
- ✅ TTS 오버랩 방지 (중복 재생 제거)
- ✅ 시작 인사 비활성화 (Wakeword 후 시작)
- ✅ 단일 언어 응답 강제 (JARVIS_LANG)
- ✅ STT 민감도 환경 변수 지원

## 라이선스

이 프로젝트는 DUM-E 프로젝트의 일부입니다.
