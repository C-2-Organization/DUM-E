# webcam/routes/__init__.py
from .situation import router as situation_router
from .monitor import router as monitor_router

__all__ = [
    "situation_router",
    "monitor_router",
]
