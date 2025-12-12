# webcam/main.py
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())  # 현재 경로 기준으로 부모 폴더들까지 .env 탐색해서 로드


from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from webcam.routes import situation_router
from webcam.workers import start_worker_and_camera


webcam = FastAPI(
    title="Dum-E Situation Server",
    version="0.1.0",
)

# CORS 필요하면 허용 (프론트에서 호출할 때)
webcam.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 필요하면 도메인 제한
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@webcam.on_event("startup")
def on_startup():
    # 실시간 카메라 + GPT 워커 시작
    start_worker_and_camera()


# 라우터 등록
webcam.include_router(situation_router)
