from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict

# ===== 0. sys.path에 상위 디렉토리(services)를 추가 =====
# 현재 파일: DUM-E/services/llm_agent/app/skill_planner.py
# env_loader: DUM-E/services/common/env_loader.py

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))          # .../services/llm_agent/app
LLM_AGENT_DIR = os.path.dirname(CURRENT_DIR)                      # .../services/llm_agent
SERVICES_ROOT = os.path.dirname(LLM_AGENT_DIR)                    # .../services

if SERVICES_ROOT not in sys.path:
    sys.path.insert(0, SERVICES_ROOT)

from common.env_loader import load_env, get_env  # noqa: E402

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser


# ===== 1. 환경 변수 로드 & LLM 초기화 =====

load_env()
API_KEY = get_env("OPENAI_API_KEY")

_base_llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0.0,
    api_key=API_KEY,
)

# JSON 객체 강제
llm = _base_llm.bind(response_format={"type": "json_object"})
parser = JsonOutputParser()


# ===== 2. 시스템 프롬프트 텍스트 =====

SYSTEM_PROMPT_TEXT = """
당신은 데스크 협동로봇 DUM-E를 위한 "스킬 플로우 플래너"입니다.

역할:
- 사용자의 자연어 명령(주로 한국어)을 읽고, 로봇이 실행해야 하는 스킬(step)들의 순서와 파라미터를 설계합니다.
- 출력은 항상 JSON 형식이어야 하며, 추가 설명 문장은 절대 출력하지 않습니다.

중요한 규칙:
1. 출력은 반드시 하나의 JSON 객체여야 합니다. JSON 밖에 다른 텍스트를 넣지 마세요.
2. JSON의 필드는 다음과 같습니다:

{
  "can_execute_now": boolean,
  "reason": string,
  "steps": [
    {
      "id": string,
      "skill": string,
      "object": {
        "raw": string | null,
        "canonical_en": string | null
      },
      "params": object
    }
  ],
  "missing_skills": [
    {
      "skill": string,
      "description": string
    }
  ],
  "user_message": string
}

3. 현재 실제로 구현된 스킬은 다음과 같습니다:
   - "PICK": 책상 위에서 특정 물체를 집는다.
     - 요구 파라미터:
       - object.raw: 사용자가 말한 물체 이름 (예: "가위")
       - object.canonical_en: 인식 모델에 넘길 영어 이름 (예: "scissors")
     - params는 일단 비워둡니다: {}

4. 그 외 스킬 이름("OPEN_DRAWER", "PLACE", "PLACE_IN_DRAWER", "MOVE_TO_LOCATION" 등)은
   아직 구현되지 않았지만, "이상적인 플로우"를 설계할 때 자유롭게 사용할 수 있습니다.
   - 다만, 그런 스킬이 하나라도 포함된 경우 can_execute_now는 false이어야 합니다.
   - 이때 missing_skills에 어떤 스킬이 왜 필요한지 적어주세요.

5. open vocabulary 객체:
   - object.raw는 사용자가 말한 표현 그대로 적습니다. (예: "가위", "노란 공", "초록색 컵")
   - object.canonical_en은 영어로 된 간단한 객체 이름으로 변환합니다. (예: "scissors", "yellow ball", "green cup")
   - perception 모델이 인식 가능하도록 최대한 일반적인 영어 단어를 사용하세요.
   - 만약 적절한 영어 표현을 확신하기 어렵다면, raw를 그대로 영어 알파벳으로 표기하거나 null로 둘 수 있습니다.

6. 명령이 너무 모호하거나, 현재 스킬 셋으로는 전혀 대응할 수 없으면:
   - can_execute_now: false
   - steps: 이상적인 플로우를 설계해도 되고, 아예 빈 리스트로 둘 수도 있습니다.
   - missing_skills에 어떤 스킬이 필요할지 제안하세요.
   - user_message에는 "수행할 수 없는 명령입니다. 현재 구현된 스킬로는 XXX를 할 수 없습니다." 와 같이 솔직하게 설명하고, 필요한 스킬을 함께 알려주세요.

7. 간단한 명령의 예:
   - "가위 잡아", "가위를 집어줘", "책상 위 가위 좀 들어" 등은 모두 PICK 하나로 처리할 수 있습니다.
     - can_execute_now: true
     - steps: [ PICK + object_name ]
     - missing_skills: []

8. 복합 명령의 예:
   - "가위를 서랍에 넣어줘" 같은 경우:
     - 이상적인 steps 예시:
       1) OPEN_DRAWER
       2) PICK "scissors"
       3) PLACE_IN_DRAWER "scissors"
       4) CLOSE_DRAWER
     - 하지만 현재는 PICK만 구현되어 있으므로:
       - can_execute_now: false
       - missing_skills: OPEN_DRAWER, PLACE_IN_DRAWER, CLOSE_DRAWER 등을 나열
       - user_message에서 이런 스킬이 필요함을 설명

반드시 이 포맷을 지키고, JSON 이외의 텍스트는 출력하지 마세요.
"""

# 여기서만 템플릿 변수 사용: {system_prompt}, {input}
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "{system_prompt}"),
        ("user", "{input}"),
    ]
)

# prompt → llm → JSON parser
chain = prompt | llm | parser


# ===== 3. 외부에서 사용할 함수 =====

def plan_skill_flow(command_text: str) -> Dict[str, Any]:
    """
    자연어 명령을 받아서 스킬 플로우(JSON dict)를 반환.
    """
    result: Dict[str, Any] = chain.invoke(
        {
            "system_prompt": SYSTEM_PROMPT_TEXT,
            "input": command_text,
        }
    )
    return result


# ===== 4. 간단 테스트용 메인 =====

if __name__ == "__main__":
    test_commands = [
        "가위 잡아",
        "가위를 서랍에 넣어줘",
        "책상을 정리해줘",
    ]

    for cmd in test_commands:
        print("=" * 80)
        print(f"INPUT: {cmd}")
        result = plan_skill_flow(cmd)
        print("OUTPUT JSON:")
        print(json.dumps(result, ensure_ascii=False, indent=2))
