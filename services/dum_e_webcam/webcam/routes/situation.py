# webcam/routes/situation.py

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

import cv2
import numpy as np

from webcam.services import analyze_situation

router = APIRouter(
    prefix="/situation",
    tags=["situation"],
)


def _file_to_frame(file_bytes: bytes):
    """업로드된 이미지 파일을 OpenCV frame으로 변환"""
    nparr = np.frombuffer(file_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return frame


@router.post("/analyze")
async def analyze_situation_api(file: UploadFile = File(...)):
    """
    이미지 1장을 업로드하면 GPT-4o로 상황 판단 결과(JSON)를 바로 돌려주는 엔드포인트.
    """
    if file.content_type not in ("image/jpeg", "image/png"):
        raise HTTPException(
            status_code=400, detail="JPEG 또는 PNG 이미지 파일만 지원합니다."
        )

    file_bytes = await file.read()
    frame = _file_to_frame(file_bytes)

    if frame is None:
        raise HTTPException(status_code=400, detail="이미지 디코딩 실패")

    try:
        result = analyze_situation(frame)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"GPT 상황 분석 중 오류: {e}")

    return JSONResponse(content=result)
