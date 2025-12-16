# services/audio_io/app/tts_streaming.py
"""
PCM 스트리밍 기반 TTS
- OpenAI TTS로부터 PCM 샘플을 실시간 수신
- 동시에:
  1) 스피커 출력
  2) UI로 파형 데이터 전송 (콜백)
"""

import io
import tempfile
import threading
from typing import Optional, Callable
import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wav
from openai import OpenAI
from services.common.env_loader import load_env, get_env


class StreamingTTS:
    """
    PCM 스트리밍 기반 TTS
    - speak() 호출 시 OpenAI TTS API로부터 WAV 수신
    - 실시간으로 PCM 샘플을 스피커 출력 + UI 콜백 전달
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini-tts",
        voice: str = "onyx",
        effect: str = "jarvis",
        sample_rate: int = 24000,  # OpenAI TTS 기본 샘플레이트
        chunk_size: int = 4096,     # UI 갱신 단위
    ):
        load_env()
        api_key = get_env("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not found in .env")

        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.voice = voice
        self.effect = effect
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size

        # 동시 재생 방지 및 중도 정지를 위한 동기화 객체
        self._play_lock = threading.RLock()
        self._stop_event = threading.Event()

        # UI 콜백 (선택)
        self.on_audio_chunk: Optional[Callable[[np.ndarray, int], None]] = None
        self.on_speaking_start: Optional[Callable[[], None]] = None
        self.on_speaking_end: Optional[Callable[[], None]] = None

        print(
            f"[StreamingTTS] 초기화: model={self.model}, voice={self.voice}, "
            f"effect={self.effect}, sr={self.sample_rate}"
        )

    def set_audio_callback(self, callback: Callable[[np.ndarray, int], None]):
        """
        UI에서 파형/RMS 표시를 위한 콜백 등록
        callback(pcm_samples: np.ndarray, sample_rate: int)
        """
        self.on_audio_chunk = callback

    def set_speaking_callbacks(
        self,
        on_start: Optional[Callable[[], None]] = None,
        on_end: Optional[Callable[[], None]] = None,
    ):
        """
        말하기 시작/종료 이벤트 콜백
        """
        self.on_speaking_start = on_start
        self.on_speaking_end = on_end

    def _apply_jarvis_effect(self, data: np.ndarray, sr: int) -> np.ndarray:
        """
        자비스 효과 (기존 tts.py에서 가져옴)
        """
        x = data.astype(np.float32)

        # FFT 기반 EQ
        X = np.fft.rfft(x, axis=0)
        n = X.shape[0]

        low = int(n * 0.05)
        mid = int(n * 0.25)
        high = int(n * 0.55)

        X[:low] *= 1.2
        X[low:mid] *= 1.10
        X[high:] *= 0.65

        y = np.fft.irfft(X, n=data.shape[0], axis=0)

        # Soft clipping + bit-crush
        y = y / 32768.0
        y = np.tanh(y * 1.4)
        levels = 512.0
        y = np.round(y * levels) / levels
        y = y * 32768.0

        # Metallic chorus / delays
        num_samples = y.shape[0]
        y_mix = y.copy()

        delays_ms = [4, 9, 13]
        gains = [0.18, 0.12, 0.08]

        for d_ms, g in zip(delays_ms, gains):
            d_samples = int(sr * d_ms / 1000.0)
            if 0 < d_samples < num_samples:
                delayed = np.zeros_like(y_mix)
                delayed[d_samples:] = y[:-d_samples] * g
                y_mix += delayed

        y_mix *= 0.85
        return np.clip(y_mix, -32768, 32767).astype(np.int16)

    def speak(self, text: str) -> bool:
        """
        텍스트를 TTS로 변환하고 스트리밍 재생
        동시에 UI 콜백 호출
        """
        text = text.strip()
        if not text:
            print("[StreamingTTS] 빈 텍스트, 건너뜀")
            return False

        print(f"[StreamingTTS] 시작: '{text[:50]}...' (voice={self.voice})")

        # 이전 재생 중이면 중단 신호
        self.stop()

        # 동시 재생 방지
        with self._play_lock:
            # 재생 시작 준비
            self._stop_event.clear()

            # 말하기 시작 이벤트
            if self.on_speaking_start:
                self.on_speaking_start()

            try:
                # 임시 파일로 WAV 저장
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    tmp_path = tmp.name

                with self.client.audio.speech.with_streaming_response.create(
                    model=self.model,
                    voice=self.voice,
                    input=text,
                    response_format="wav",
                ) as response:
                    response.stream_to_file(tmp_path)

                # WAV 로드
                sr, data = wav.read(tmp_path)

                # 효과 적용
                if self.effect == "jarvis":
                    data = self._apply_jarvis_effect(data, sr)
                    playback_sr = int(sr * 0.95)  # 약간 느리게
                else:
                    playback_sr = sr

                # 청크 단위로 재생 + UI 콜백
                self._play_with_callback(data, playback_sr)

                print("[StreamingTTS] 재생 완료")
                return True

            except Exception as e:
                print(f"[StreamingTTS] 오류: {e}")
                return False
            finally:
                # 말하기 종료 이벤트
                if self.on_speaking_end:
                    self.on_speaking_end()

    def _play_with_callback(self, data: np.ndarray, sr: int):
        """
        PCM 데이터를 청크 단위로 재생하며 UI 콜백 호출
        """
        total_samples = len(data)
        chunk_size = self.chunk_size

        # sounddevice 스트림 열기
        stream = sd.OutputStream(
            samplerate=sr,
            channels=1,
            dtype='int16',
            blocksize=chunk_size,
        )
        stream.start()

        try:
            for start in range(0, total_samples, chunk_size):
                if self._stop_event.is_set():
                    break
                end = min(start + chunk_size, total_samples)
                chunk = data[start:end]

                # 스피커 출력
                stream.write(chunk)

                # UI 콜백
                if self.on_audio_chunk:
                    self.on_audio_chunk(chunk, sr)

        finally:
            stream.stop()
            stream.close()

    def stop(self):
        """현재 재생 중인 오디오를 중단 요청"""
        self._stop_event.set()
