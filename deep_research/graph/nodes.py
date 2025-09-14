from typing import Literal

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.constants import END
from langgraph.prebuilt.chat_agent_executor import AgentState
from langgraph.runtime import Runtime
from langgraph.types import Command

from common.log import logger
from deep_research.graph.context import StaticContext
from deep_research.graph.prompts import final_report_generation_prompt
from deep_research.graph.state import DeepResearchState, ClarifyWithUser, ResearchQuestion
from deep_research.graph.supervisor.prompts import lead_researcher_prompt
from deep_research.llms.llm import get_llm
from deep_research.prompts.deep_research_prompts import clarify_with_user_instructions_prompt, \
    transform_messages_into_research_topic_prompt
from deep_research.utils.utils import now, get_buffer_string


async def clarify_with_user(state: DeepResearchState, runtime: Runtime[StaticContext]) -> Command[
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
    # 如果禁止人在环路，则直接进入深度研究流程
    static_context = runtime.context
    if not static_context.allow_clarification:
        return Command(
            goto="write_research_brief",  # 撰写研究简介流程
        )

    # 获取用户研究主题
    messages = state["messages"]

    # 定义用户澄清模型实例 + 结构化输出
    clarification_model = get_llm(
        model=static_context.research_model,
        max_tokens=static_context.research_model_max_tokens
    ).with_structured_output(ClarifyWithUser)  # 结构化输出

    # 用户澄清提示词
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


async def write_research_brief(state: DeepResearchState, runtime: Runtime[StaticContext]) -> Command[
    Literal["research_supervisor"]]:
    """
    将用户问题 和 澄清结果信息 转换为一个结构化的研究简介实体，然后提供给研究主管智能体去开始进行深度研究

    分析用户消息并生成一个聚焦的研究简介，用于指导研究主管智能体。
    同时设置初始的研究主管上下文，包含适当的提示和指令。
    """

    static_context = runtime.context

    # 定义 研究规划模型实例 + 结构化输出
    research_model = get_llm(
        model=static_context.research_model,
        max_tokens=static_context.research_model_max_tokens
    ).with_structured_output(ResearchQuestion)  # 获取结构化输出

    # 生成研究规划简介提示词
    prompt = transform_messages_into_research_topic_prompt.format(
        messages=get_buffer_string(state.get("messages", [])),
        date_time=now(),
    )

    # 获取深度研究规划简介
    response = await research_model.ainvoke([HumanMessage(content=prompt)])

    # 使用研究简介和指令初始化主管智能体
    # 设置主管节点的系统提示词
    supervisor_system_prompt = lead_researcher_prompt.format(
        date_time=now(),
        max_concurrent_research_units=static_context.max_concurrent_research_units,
        max_researcher_iterations=static_context.max_researcher_iterations,
    )

    return Command(
        goto="research_supervisor",
        update={
            "research_brief": response.research_brief,
            "supervisor_messages": {
                "type": "override",
                "value": [
                    SystemMessage(content=supervisor_system_prompt),
                    HumanMessage(content=response.research_brief),
                ]
            }
        }

    )


async def final_report_generation(state: DeepResearchState, runtime: Runtime[StaticContext]):
    """
        生成最终的综合研究报告，并包含针对令牌限制的重试逻辑。
        该函数接收所有收集到的研究发现，并使用配置的报告生成模型将其综合成一个结构良好、内容全面的最终报告。

        参数:
            state: 包含研究发现和上下文的代理状态
            runtime: 静态运行上下文

        返回:
            包含最终报告和已清理状态的字典
        """

    static_context = runtime.context

    # 获取深度研究成果
    notes = state.get("notes", [])
    # 清理状态机
    cleared_state = {"notes": {"type": "override", "value": []}}
    # 组装为一个大的字符串
    findings = "\n".join(notes)

    # 撰写最终的研究报告
    final_report_model = get_llm(
        model=static_context.final_report_model,
        max_tokens=static_context.final_report_model_max_tokens
    )

    final_report_prompt = final_report_generation_prompt.format(
        date_time=now(),
        research_brief=state.get("research_brief", ""),
        messages=get_buffer_string(state.get("messages", [])),
        findings=findings,
    )

    final_report = await final_report_model.ainvoke([HumanMessage(content=final_report_prompt)])

    # Return successful report generation
    return {
        "final_report": final_report.content,
        "messages": [final_report],
        **cleared_state
    }
