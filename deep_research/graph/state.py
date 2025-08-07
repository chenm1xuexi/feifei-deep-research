from typing import Annotated, Optional

from langchain_core.messages import MessageLikeRepresentation
from langgraph.graph import MessagesState
from pydantic import BaseModel, Field

from deep_research.utils import override_reducer


class DeepResearchState(MessagesState):
    """ 深度研究state, 主agent 状态上下文 """
    # 主管智能体 消息上下文
    supervisor_messages: Annotated[list[MessageLikeRepresentation], override_reducer]
    # 研究摘要，通过人在环路后，交由大模型生成的研究主题摘要，是后续进行深度研究的核心基础
    research_brief: Optional[str]
    raw_notes: Annotated[list[str], override_reducer] = []
    notes: Annotated[list[str], override_reducer] = []
    # 最终生成的研究报告
    final_report: str


class ClarifyWithUser(BaseModel):
    """ 是否需要进行用户澄清的实体， 由大模型返回 """
    need_clarification: bool = Field(
        description="Whether the user needs to be asked a clarifying question."
    )

    question: str = Field(
        description="A question to ask the user to clarify the report scope."
    )

    verification: str = Field(
        description="Verify message that we will start research after the user has provided the necessary information."
    )


class ResearchQuestion(BaseModel):
    """ 研究问题和指导研究的简介 """
    research_brief: str = Field(
        description="A research question that will be used to guide the research."
    )
