from typing import TypedDict, Annotated

from langchain_core.messages import MessageLikeRepresentation
from pydantic import BaseModel, Field

from deep_research.utils import override_reducer


class SupervisorState(TypedDict):
    """ 主管智能体 上下文 """

    # 主管智能体消息列表
    supervisor_messages: Annotated[list[MessageLikeRepresentation], override_reducer]
    # 研究规划简介
    research_brief: str
    # 深度研究迭代次数
    research_iterations: int = 0
    # 研究成果列表 经过过滤后的
    notes: Annotated[list[str], override_reducer] = []
    # 研究原始成果列表
    raw_notes: Annotated[list[str], override_reducer] = []


class ConductResearch(BaseModel):
    """Call this tool to conduct research on a specific topic."""

    # 简单来说 这个topic就是拆解的子任务，然后分配给具体的子智能体进行深度研究
    research_topic: str = Field(
        description="The topic to research. Should be a single topic, and should be described in high detail (at least a paragraph).",
    )


class ResearchComplete(BaseModel):
    """Call this tool to indicate that the research is complete."""
    # 当主管智能体，认定研究结果以满足回答用户问题时，将调用此工具来标识深入研究完成
