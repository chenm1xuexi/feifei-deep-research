from typing import Optional

from langchain_deepseek import ChatDeepSeek
import os
from dotenv import load_dotenv

load_dotenv()


def get_llm(
        model,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        max_tokens=10000,
) -> ChatDeepSeek:
    """ 获取 llm 实例"""
    if not base_url:
        base_url = os.getenv("LLM_BASE_URL")
    if not api_key:
        api_key = os.getenv("LLM_API_KEY")
    return ChatDeepSeek(
        api_base=base_url,
        model=model,
        api_key=api_key,
        max_tokens=max_tokens,
    )
