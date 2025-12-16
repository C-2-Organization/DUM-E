# services/audio_io/app/jarvis_hud.py
"""
Jarvis 스타일 HUD (Head-Up Display)
- PySide6 기반 네이티브 윈도우
- GIF 애니메이션 배경
- PCM 샘플 기반 자연스러운 애니메이션
"""

import sys
import os
import numpy as np
from collections import deque
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget
from PySide6.QtCore import Qt, QTimer, Signal, Slot
from PySide6.QtGui import QPainter, QColor, QFont, QMovie, QRadialGradient, QPen


class JarvisHUD(QMainWindow):
    """
    Jarvis 관리 HUD 윈도우
    - 중앙: GIF 애니메이션
    - 상태: idle / listening / thinking / speaking
    """

    # 시그널 정의 (스레드 안전)
    audio_chunk_signal = Signal(np.ndarray, int)
    state_changed_signal = Signal(str)

    def __init__(self):
        super().__init__()

        # 윈도우 설정
        self.setWindowTitle("JARVIS Management Console")
        self.setGeometry(100, 100, 800, 600)
        self.setStyleSheet("background-color: #0a0a0a;")

        # 중앙 위젯
        self.central_widget = HUDWidget()
        self.setCentralWidget(self.central_widget)

        # 시그널 연결
        self.audio_chunk_signal.connect(self.central_widget.on_audio_chunk)
        self.state_changed_signal.connect(self.central_widget.on_state_changed)

    @Slot(np.ndarray, int)
    def on_audio_chunk(self, samples: np.ndarray, sample_rate: int):
        """
        외부(TTS)에서 호출: PCM 샘플 수신
        """
        self.audio_chunk_signal.emit(samples, sample_rate)

    @Slot(str)
    def set_state(self, state: str):
        """
        외부에서 호출: 상태 변경 (idle/listening/thinking/speaking)
        """
        self.state_changed_signal.emit(state)


class HUDWidget(QWidget):
    """
    실제 렌더링 위젯 - GIF 애니메이션 배경
    """

    def __init__(self):
        super().__init__()

        # 상태
        self.state = "idle"  # idle / listening / thinking / speaking

        # 파형 버퍼 (최근 N개 샘플)
        self.waveform_buffer_inner = deque(maxlen=200)
        self.rms_level = 0.0

        # 애니메이션
        self.glow_intensity = 0.0

        # 말하기 효과 애니메이션용
        self.effect_phase = 0.0
        self.effect_timer = QTimer(self)
        self.effect_timer.setInterval(33)  # ~30 FPS
        self.effect_timer.timeout.connect(self._on_effect_tick)

        # GIF 로딩
        gif_path = os.path.join(os.path.dirname(__file__), "assets", "jarvis.gif")
        self.movie = QMovie(gif_path)
        self.current_pixmap = None
        
        # GIF 프레임 변경 시그널 연결 (GIF 자체 프레임레이트 사용)
        self.movie.frameChanged.connect(self._update_frame)
        self.movie.setCacheMode(QMovie.CacheAll)  # 전체 캐싱으로 부드럽게
        self.movie.setSpeed(100)  # 100% 속도
        self.movie.start()

    def _update_frame(self):
        """GIF 프레임 업데이트"""
        self.current_pixmap = self.movie.currentPixmap()
        self.update()

    def _on_effect_tick(self):
        """말하기 효과 애니메이션 틱"""
        self.effect_phase = (self.effect_phase + 0.15) % (2 * np.pi)
        self.update()

    @Slot(np.ndarray, int)
    def on_audio_chunk(self, samples: np.ndarray, sample_rate: int):
        """
        PCM 샘플 수신 → 파형 버퍼 갱신
        """
        # int16 → float32 정규화
        norm = samples.astype(np.float32) / 32768.0

        # RMS 계산
        rms = np.sqrt(np.mean(norm ** 2))
        self.rms_level = float(rms)

        # 다운샘플링 (화면에 맞게)
        step = max(1, len(norm) // 100)
        downsampled = norm[::step]

        # 버퍼에 추가
        for idx, val in enumerate(downsampled):
            v = float(val)
            self.waveform_buffer_inner.append(v)

    @Slot(str)
    def on_state_changed(self, state: str):
        """
        상태 변경
        """
        print(f"[HUD] 상태 변경: {self.state} → {state}")
        self.state = state

        # speaking 종료 시 파형 버퍼 클리어
        if state != "speaking":
            self.waveform_buffer_inner.clear()
            self.rms_level = 0.0

        # GIF는 항상 재생, 효과 타이머는 정지
        try:
            if self.movie.state() != QMovie.Running:
                self.movie.start()
            if self.effect_timer.isActive():
                self.effect_timer.stop()
        except Exception:
            pass

    def paintEvent(self, event):
        """
        Qt 페인트 이벤트 → 매 프레임 호출
        """
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        width = self.width()
        height = self.height()

        # 배경
        painter.fillRect(0, 0, width, height, QColor(10, 10, 10))

        # GIF 프레임 그리기 (중앙에 배치)
        if self.current_pixmap:
            pixmap = self.current_pixmap
            # 화면 크기에 맞게 스케일
            scaled_size = min(width, height)
            pixmap = pixmap.scaled(scaled_size, scaled_size, 
                                  Qt.KeepAspectRatio, Qt.SmoothTransformation)
            
            # 중앙 정렬
            x = (width - pixmap.width()) // 2
            y = (height - pixmap.height()) // 2
            painter.drawPixmap(x, y, pixmap)

            # 효과 요청 해제: GIF는 항상 그대로 재생

        # 상태 텍스트 (GIF 위에 오버레이)
        self._draw_status_text(painter, width, height)
        
        # RMS 기반 글로우 강도 업데이트
        if self.rms_level > 0.01:
            self.glow_intensity = min(1.0, self.glow_intensity + 0.15)
        else:
            self.glow_intensity = max(0.0, self.glow_intensity - 0.08)

    def _draw_speaking_effects(self, painter: QPainter, x: int, y: int, w: int, h: int, s: float):
        """자비스 발화 시 GIF 위에 발광/틴트 효과"""
        cx = x + w // 2
        cy = y + h // 2
        radius = int(min(w, h) * (0.46 + 0.06 * s))

        # 1) Bloom: Screen 합성으로 사이안 발광을 얕게 오버레이
        painter.save()
        painter.setOpacity(0.12 * s)
        painter.setCompositionMode(QPainter.CompositionMode_Screen)
        grad = QRadialGradient(cx, cy, radius)
        c1 = QColor(90, 220, 255, int(160 * s))
        c2 = QColor(0, 0, 0, 0)
        grad.setColorAt(0.0, c1)
        grad.setColorAt(0.6, QColor(90, 220, 255, int(90 * s)))
        grad.setColorAt(1.0, c2)
        painter.setBrush(grad)
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(cx - radius, cy - radius, radius * 2, radius * 2)
        painter.restore()

        # 2) Edge ring: 얇은 링으로 선명도 강조
        ring_color = QColor(120, 240, 255, int(120 * s))
        pen = QPen(ring_color)
        pen.setWidth(2)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(cx - radius, cy - radius, radius * 2, radius * 2)

    def _draw_status_text(self, painter: QPainter, width: int, height: int):
        """
        상태 텍스트 표시
        """
        font = QFont("Courier", 14, QFont.Bold)
        painter.setFont(font)
        painter.setPen(QColor(180, 180, 200))

        status_text = {
            "idle": "STANDBY",
            "listening": "LISTENING...",
            "thinking": "PROCESSING...",
            "speaking": "SPEAKING",
        }
        text = status_text.get(self.state, "UNKNOWN")

        painter.drawText(10, height - 30, text)


def main():
    """
    HUD 단독 테스트용 메인
    """
    app = QApplication(sys.argv)
    hud = JarvisHUD()
    hud.show()

    # 테스트: 3초마다 상태 변경
    def test_cycle():
        states = ["idle", "listening", "thinking", "speaking", "idle"]
        idx = [0]

        def next_state():
            idx[0] = (idx[0] + 1) % len(states)
            hud.set_state(states[idx[0]])

        test_timer = QTimer()
        test_timer.timeout.connect(next_state)
        test_timer.start(3000)

    test_cycle()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
