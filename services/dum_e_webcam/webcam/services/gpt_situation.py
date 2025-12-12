# webcam/services/gpt_situation.py
import base64
import json
from pathlib import Path

import cv2
from openai import OpenAI

# .env에 있는 OPENAI_API_KEY를 자동으로 사용
client = OpenAI()


def encode_image(frame) -> str:
    """OpenCV frame -> base64 JPEG 문자열"""
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf.tobytes()).decode("utf-8")


# situation_prompt.txt 읽기 (app 디렉터리 기준)
PROMPT_PATH = Path(__file__).resolve().parent.parent / "situation_prompt.txt"
SITUATION_PROMPT = PROMPT_PATH.read_text(encoding="utf-8")


def analyze_situation(frame):
    """
    프레임 1장을 GPT-4o-mini로 분석해서 '상황 JSON(dict)' 반환
    """
    image_b64 = encode_image(frame)

    resp = client.chat.completions.create(
        model="gpt-4o-mini",  # 필요하면 "gpt-4o"로 변경 가능
        messages=[
            {
                "role": "system",
                "content": "You are a vision-based situation understanding module. Output strict JSON only.",
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": SITUATION_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_b64}"
                        },
                    },
                ],
            },
        ],
        max_tokens=250,
        temperature=0.0,
    )

    text = resp.choices[0].message.content
    return json.loads(text)
