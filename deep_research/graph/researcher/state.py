import operator
from typing import TypedDict, Annotated

from langchain_core.messages import MessageLikeRepresentation
from pydantic import BaseModel

from deep_research.utils import override_reducer


class ResearcherState(TypedDict):
    """ 研究员 状态机， 用于存储子任务的具体研究相关信息 """

    # 研究子任务信息列表
    researcher_messages: Annotated[list[MessageLikeRepresentation], operator.add]
    # 工具调度迭代次数
    tool_call_iterations: int = 0
    # 压缩后的研究信息
    compressed_research: str
    # 研究成果
    raw_notes: Annotated[list[str], override_reducer] = []


class ResearcherOutputState(BaseModel):
    """ 单个研究员研究成果 """

    compressed_research: str
    raw_notes: Annotated[list[str], override_reducer] = []


class Summary(BaseModel):
    """
    Research summary with key findings.
    """
    # 包含关键发现的研究摘要。

    # 摘要
    summary: str
    # 引用
    key_excerpts: str






