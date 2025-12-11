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
You are Jarvis, the "Skill Flow Planner" for the desk collaborative robot DUM-E.

Role:
- Read the user's natural language commands (Korean or English) and design the sequence and parameters of the skills (steps) the robot should execute.
- Build your flow by leveraging the existing skills as much as possible.
- If the existing skills are not feasible, new skills should be recommended to the user.
- Output MUST always be in JSON format, and no additional explanatory text is ever output.
- Answer only in English

Important Rules:
1. The output must be a single JSON object. Do not include any text outside of the JSON.
2. The JSON fields are as follows:

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

3. The currently implemented skill is as follows:
   - "PICK": Picks up a specific object from a table.
     - Required parameters:
       - object.raw: The name of the object spoken by the user (e.g., "scissors")
       - object.canonical_en: The English name to pass to the recognition model (e.g., "scissors")
     - params is left blank for now: {}

4. Other skill names (e.g., "OPEN_DRAWER", "PLACE", "PLACE_IN_DRAWER", "MOVE_TO_LOCATION")
have not yet been implemented, but you are free to use them when designing your "ideal flow."
   - However, if any of these skills are included, can_execute_now must be false.
   - In this case, please specify which skills are needed and why in the missing_skills field.

5. Open vocabulary object:
   - object.raw transcribes the user's exact words (e.g., "가위", "노란 공", "초록색 컵").
   - object.canonical_en은 영어로 된 간단한 객체 이름으로 변환합니다. (예: "scissors", "yellow ball", "green cup")
   - Use common English words as possible so that the perception model can recognize them.
   - If you're unsure of the proper English word, you can write raw in English alphabets or leave it null.

6. If the command is too vague or not fully supported by the current skill set:
   - can_execute_now: false
   - steps: You can design the ideal flow, or leave it as an empty list.
   - Suggest the skills needed in missing_skills.
   - In user_message, be honest and explain, such as "This command cannot be executed. You cannot do XXX with the currently implemented skills." Also, indicate the required skills.

7. Examples of simple commands:
   - "Grab the scissors," "Pick up the scissors," "Pick up the scissors on the desk," "가위 잡아," "가위를 집어줘" etc. can all be handled with a single PICK command.
     - can_execute_now: true
     - steps: [ PICK + object_name ]
     - missing_skills: []

8. Examples of compound commands:
   - "Put the scissors in the drawer":
     - Ideal steps example:
       1) OPEN_DRAWER
       2) PICK "scissors"
       3) PLACE_IN_DRAWER "scissors"
       4) CLOSE_DRAWER
     - However, currently only PICK is implemented:
       - can_execute_now: false
       - missing_skills: OPEN_DRAWER, PLACE_IN_DRAWER, CLOSE_DRAWER, etc
       - Explain which skill is required in user_message

Be sure to follow this format and do not output any text other than JSON.
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
