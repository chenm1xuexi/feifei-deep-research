from typing import Literal

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.constants import END
from langgraph.types import Command

from deep_research.graph.configuration import Configuration
from deep_research.graph.state import DeepResearchState, ClarifyWithUser, ResearchQuestion
from deep_research.llms.llm import get_llm
from deep_research.prompts.deep_research_prompts import clarify_with_user_instructions_prompt, \
    transform_messages_into_research_topic_prompt
from deep_research.utils.utils import now, get_buffer_string

from common.log import logger


async def clarify_with_user(state: DeepResearchState, config: RunnableConfig) -> Command[
    Literal["write_research_brief", "__end__"]]:
    """
        人在环路， 分析用户消息，如果研究范围不明确则询问澄清问题。

        当前节点确定用户请求在进行研究之前是否需要澄清。
        如果禁用澄清或不需要澄清，则直接进入研究阶段。

        参数:
            state: 包含用户消息的当前代理状态
            config: 包含模型设置和首选项的运行时配置
        返回:
            结果路由，用于结束澄清问题或直接进入深度研究流程
    """

    # 获取通用配置
    configurable = Configuration.from_runnable_config(config)
    # 如果禁止人在环路，则直接进入深度研究流程
    if not configurable.allow_clarification:
        return Command("write_research_brief")

    # 获取用户研究主题
    messages = state["messages"]

    # 初始化用户确认反馈的模型调度实例
    clarification_model = (get_llm(model=configurable.clarification_model,
                                   max_tokens=configurable.clarification_model_max_tokens)
                           .with_structured_output(ClarifyWithUser)  # 结构化输出
                           .with_retry(stop_after_attempt=configurable.max_structured_output_retries)  # 失败重试策略
                           )

    # 调度模型 获取是否需要向用户进行问题信息确认的结构化输出结果
    prompt = clarify_with_user_instructions_prompt.format(
        messages=get_buffer_string(messages),
        date_time=now(),
    )

    response = await clarification_model.ainvoke([HumanMessage(content=prompt)])
    logger.info(f"clarification_model response: {response}")

    if response.need_clarification:
        # 如果需要进一步向用户确认信息，则直接结束, 等待用户反馈
        return Command(
            goto=END,
            update={"messages": [AIMessage(content=response.question)]}
        )
    else:
        # 不需要，则直接进入撰写研究简介节点
        return Command(
            goto="write_research_brief",
            update={"messages": [AIMessage(content=response.verification)]}
        )


async def write_research_brief(state: DeepResearchState, config: RunnableConfig) -> Command[
    Literal["research_supervisor"]]:
    """
    将用户问题 和 澄清结果信息 转换为一个结构化的研究简介实体，然后提供给研究主管智能体去开始进行深度研究

    分析用户消息并生成一个聚焦的研究简介，用于指导研究主管智能体。
    同时设置初始的研究主管上下文，包含适当的提示和指令。
    """

    configurable = Configuration.from_runnable_config(config)

    # 初始化 生成 研究简介 模型调度实例
    research_model = (get_llm(model=configurable.research_model,
                              max_tokens=configurable.research_model_max_tokens)
                      .with_structured_output(ResearchQuestion)  # 获取结构化输出
                      .with_retry(stop_after_attempt=configurable.max_structured_output_retries)  # 失败重试策略
                      )

    prompt = transform_messages_into_research_topic_prompt.format(
        messages=get_buffer_string(state.get("messages", [])),
        date_time=now(),
    )

    # 生成研究简介 + 指导（其实这里很类似生成研究计划）
    response = await research_model.ainvoke([HumanMessage(content=prompt)])

    # 设置主管节点的系统提示词
    supervisor_system_prompt = """
    
    """
