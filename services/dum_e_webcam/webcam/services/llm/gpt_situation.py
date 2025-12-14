# webcam/services/llm/gpt_situation.py
import base64
import json
from pathlib import Path

import cv2
from openai import OpenAI

client = OpenAI()

# ✅ prompt 파일은 한 곳에만 두는 걸 추천: webcam/situation_prompt.txt
PROMPT_PATH = Path("/home/ilhoon/DUM-E/services/dum_e_webcam/webcam/situation_prompt.txt")

_PROMPT_CACHE = None


def _load_prompt() -> str:
    global _PROMPT_CACHE
    if _PROMPT_CACHE is not None:
        return _PROMPT_CACHE

    if not PROMPT_PATH.exists():
        # 서버 죽지 말고 기본 프롬프트로라도 동작
        print(f"[gpt_situation] WARN: prompt not found: {PROMPT_PATH}")
        _PROMPT_CACHE = "You are a vision module. Output strict JSON only."
        return _PROMPT_CACHE

    _PROMPT_CACHE = PROMPT_PATH.read_text(encoding="utf-8")
    return _PROMPT_CACHE


def encode_image(frame) -> str:
    """OpenCV frame -> base64 JPEG 문자열"""
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def analyze_situation(frame, meta=None):
    """
    프레임 1장을 GPT-4o-mini로 분석해서 '상황 JSON(dict)' 반환
    meta가 있으면 prompt.txt의 {placeholders}를 채워 넣음
    """
    if frame is None or not hasattr(frame, "shape"):
        raise ValueError(f"Invalid frame type: {type(frame)}")

    image_b64 = encode_image(frame)

    prompt_template = _load_prompt()

    in_table_roi = meta.get("in_table_roi") if isinstance(meta, dict) else None
    between_holes = meta.get("between_holes") if isinstance(meta, dict) else None
    robot_target_xy = meta.get("robot_target_xy") if isinstance(meta, dict) else None

    if isinstance(meta, dict):
        in_table_roi = str(meta.get("in_table_roi", "unknown"))
        between_holes = str(meta.get("between_holes", "unknown"))
        robot_target_xy = str(meta.get("robot_target_xy", "unknown"))

    prompt = prompt_template
    prompt = prompt.replace("{in_table_roi}", str(in_table_roi))
    prompt = prompt.replace("{between_holes}", str(between_holes))
    prompt = prompt.replace("{robot_target_xy}", str(robot_target_xy))


    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a vision-based situation understanding module. Output strict JSON only.",
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                    },
                ],
            },
        ],
        max_tokens=250,
        temperature=0.0,
    )

    text = resp.choices[0].message.content
    return json.loads(text)
