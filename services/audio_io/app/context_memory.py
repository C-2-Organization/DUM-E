# services/audio_io/app/context_memory.py
from __future__ import annotations
from collections import deque
from dataclasses import dataclass, field
from threading import Lock
from typing import Deque, List

@dataclass
class ContextMemory:
    maxlen: int = 5
    _buf: Deque[str] = field(default_factory=lambda: deque(maxlen=5))
    _lock: Lock = field(default_factory=Lock)

    def push(self, s: str) -> None:
        s = (s or "").strip()
        if not s:
            return
        with self._lock:
            self._buf.append(s)

    def snapshot(self) -> List[str]:
        with self._lock:
            return list(self._buf)

    def clear(self) -> None:
        with self._lock:
            self._buf.clear()
