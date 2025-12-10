# services/llm_agent/app/llm.py

from typing import Optional

from langchain_openai import ChatOpenAI  # pip install langchain-openai
from langchain_core.messages import SystemMessage, HumanMessage  # pip install langchain-core

from services.common.env_loader import load_env, get_env


# 공통 .env 로드
load_env()
API_KEY = get_env("OPENAI_API_KEY")


def _get_llm(model: str = "gpt-4o-mini", temperature: float = 0.3) -> ChatOpenAI:
    """
    LangChain ChatOpenAI 인스턴스를 만든다.
    LangChain 공식 가이드는 langchain-openai 패키지의 ChatOpenAI 사용을 권장함. :contentReference[oaicite:0]{index=0}
    """
    if not API_KEY:
        raise RuntimeError("OPENAI_API_KEY not found in .env")

    return ChatOpenAI(
        model=model,
        temperature=temperature,
        api_key=API_KEY,
    )


_llm_singleton: Optional[ChatOpenAI] = None


def get_llm() -> ChatOpenAI:
    global _llm_singleton
    if _llm_singleton is None:
        _llm_singleton = _get_llm()
    return _llm_singleton


def ask_llm(user_text: str) -> str:
    """
    STT로 받은 사용자의 발화를 LLM에 보내고, 답변 텍스트만 반환.
    """
    user_text = user_text.strip()
    if not user_text:
        return "아무 말씀도 못 들었어요. 다시 한번 말씀해 주세요."

    llm = get_llm()

    messages = [
        SystemMessage(
            content=(
                "너는 사무실에서 일을 도와주는 협동로봇 Dummy의 두뇌야. "
                "답변은 짧고 명확하게, 주로 한국어로 말해. "
                "너무 장황하게 설명하지 말고, 한두 문장 위주로 답해."
            )
        ),
        HumanMessage(content=user_text),
    ]

    result = llm.invoke(messages)
    # result는 AIMessage, content에 문자열 또는 리스트가 들어있을 수 있음
    content = result.content
    if isinstance(content, list):
        # 안전하게 string 부분만 합치기
        text_chunks = [c for c in content if isinstance(c, str)]
        return "\n".join(text_chunks) if text_chunks else str(content)
    return str(content)
