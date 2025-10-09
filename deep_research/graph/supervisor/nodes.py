
"""
主管智能体 相关节点
"""
import asyncio
from typing import Literal

from langchain_core.messages import ToolMessage, HumanMessage
from langgraph.constants import END
from langgraph.runtime import Runtime
from langgraph.types import Command, Send

from deep_research.graph.context import StaticContext
from deep_research.graph.supervisor.state import SupervisorState, ConductResearch, ResearchComplete
from deep_research.graph.tools import think_tool, get_notes_from_tool_calls
from deep_research.llms.llm import get_llm
from deep_research.graph.researcher.builder import researcher_subgraph
from langchain_core.messages.utils import trim_messages, count_tokens_approximately

from common.log import logger


async def supervisor(state: SupervisorState, runtime: Runtime[StaticContext]) -> Command[Literal["supervisor_tools"]]:
    """负责研究的主管智能体，制定研究策略并委派任务给研究人员。

        主管智能体会分析研究简报，并决定如何将研究任务分解为可管理的小任务。
        它可以使用 think_tool 进行战略规划，使用 ConductResearch 将任务委派给子研究人员，
        或在对研究结果满意时使用 ResearchComplete。

        参数:
            state: 包含消息和研究上下文的当前主管智能体状态
            runtime: 静态运行时上下文

        返回:
            命令，继续执行 supervisor_tools 工具
        """
    logger.info(f"====== supervisor start ======")
    static_context = runtime.context

    supervisor_tools = [
        ConductResearch,
        ResearchComplete,
        think_tool,
    ]

    # 定义深度研究模型实例
    research_model = get_llm(
        model=static_context.research_model,
        max_tokens=static_context.research_model_max_tokens,
    ).bind_tools(supervisor_tools)

    # 获取整体研究消息上下文
    supervisor_messages = state.get("supervisor_messages", [])

    # 因此这里也需要进行上下文压缩，因为多次研究后，可能研究深度已达最大模型上下文上限
    supervisor_messages = trim_messages(
        messages=supervisor_messages,
        strategy="last",
        max_tokens=static_context.research_model_max_context_length,
        token_counter=count_tokens_approximately,
        include_system=True,
        end_on=("human", "tool"),
    )

    response = await research_model.ainvoke(supervisor_messages)

    result = Command(
        goto="supervisor_tools",  # 直接路由到主管工具节点，完成工具执行
        update={
            "supervisor_messages": [response],  # 将主管的回复添加到消息状态中
            "research_iterations": state.get("research_iterations", 0) + 1  # 更新研究迭代次数
        }
    )

    logger.info(f"====== supervisor end ======")
    return result


async def supervisor_tools(state: SupervisorState, runtime: Runtime[StaticContext]) -> Command[
    Literal["supervisor", "__end__"]]:
    """执行主管智能体调用的工具，包括研究委派和战略思考。

        该函数处理三种类型的主管工具调用：
        1. think_tool - 战略性思考，继续对话
        2. ConductResearch - 将研究任务委派给子研究人员
        3. ResearchComplete - 标志着研究阶段的完成

        参数:
            state: 包含消息和迭代次数的当前主管状态
            runtime: 静态运行时上下文

        返回:
            命令，用于继续主管循环或结束研究阶段
        """
    logger.info(f"====== supervisor_tools start ======")

    static_context = runtime.context
    supervisor_messages = state.get("supervisor_messages", [])
    research_iterations = state.get("research_iterations", 0)

    # 最近的消息，也就是主管智能体的研究决策结果
    most_recent_message = supervisor_messages[-1]

    # 检查是否已超过允许的最大研究次数
    exceeded_allowed_iterations = research_iterations > static_context.max_researcher_iterations

    # 判定是否存在工具调用
    no_tool_calls = not most_recent_message.tool_calls

    # 判定是否包含研究完成工具调用
    research_complete_tool_call = any(
        tool_call["name"] == "ResearchComplete"
        for tool_call in most_recent_message.tool_calls
    )

    # 退出条件：超出最大研究次数 或者 不存在工具调用 或者 研究完成，则退出当前深度研究图
    if exceeded_allowed_iterations or no_tool_calls or research_complete_tool_call:
        result = Command(
            goto=END,
            update={
                "notes": get_notes_from_tool_calls(supervisor_messages),  # 提取所有研究成果的工具结果信息
                "research_brief": state.get("research_brief", ""),
            }
        )
        logger.info(f"====== supervisor_tools end (going to END) ======")
        return result

    # 处理工具调用
    all_tool_messages = []
    update_payload = {"supervisor_messages": []}

    # 判定是否存在研究策略工具调用
    think_tool_calls = [
        tool_call for tool_call in most_recent_message.tool_calls
        if tool_call["name"] == "think_tool"
    ]

    for tool_call in think_tool_calls:
        reflection_content = tool_call["args"]["reflection"]
        all_tool_messages.append(ToolMessage(
            content=f"Reflection recorded: {reflection_content}",  # 注意这里只是提取了模型的tool_call，并非调度了工具
            name="think_tool",
            tool_call_id=tool_call["id"]
        ))

    # 判定是否存在分配子任务给研究智能体的工具调用
    conduct_research_calls = [
        tool_call for tool_call in most_recent_message.tool_calls
        if tool_call["name"] == "ConductResearch"
    ]

    if conduct_research_calls:
        # 这是一种兜底策略，这里这么做的目的是防止模型分配的子任务过多，导致并发调度达到资源阈值，进而任务失败
        allowed_conduct_research_calls = conduct_research_calls[:static_context.max_concurrent_research_units]
        overflow_conduct_research_calls = conduct_research_calls[static_context.max_concurrent_research_units:]

        research_tasks = [
            researcher_subgraph.ainvoke(
                input={
                    "researcher_messages": [
                        HumanMessage(content=tool_call["args"]["research_topic"])
                    ],
                    "research_topic": tool_call["args"]["research_topic"],
                },
                context=static_context,
                subgraphs=True,
            ) for tool_call in allowed_conduct_research_calls
        ]

        # 并行调度子任务，获取子智能体 工具返回结果
        tool_results = await asyncio.gather(*research_tasks)

        # 将子智能体的研究成果组装为工具消息返回
        for observation, tool_call in zip(tool_results, allowed_conduct_research_calls):
            all_tool_messages.append(ToolMessage(
                content=observation.get("compressed_research",
                                        "Error synthesizing research report: Maximum retries exceeded"),
                name=tool_call["name"],
                tool_call_id=tool_call["id"]
            ))

        # 并发调度溢出的工具，不会进行研究，直接返回错误信息
        for overflow_call in overflow_conduct_research_calls:
            all_tool_messages.append(ToolMessage(
                content=f"Error: Did not run this research as you have already exceeded the maximum number of concurrent research units. Please try again with {static_context.max_concurrent_research_units} or fewer research units.",
                name="ConductResearch",
                tool_call_id=overflow_call["id"]
            ))

        # 聚合所有工具的研究成果为一个大的字符串
        raw_notes_concat = "\n".join([
            "\n".join(observation.get("raw_notes", []))
            for observation in tool_results
        ])

        if raw_notes_concat:
            update_payload["raw_notes"] = [raw_notes_concat]

    # 将所有的工具调用结果添加到supervisor_messages中
    update_payload["supervisor_messages"] = all_tool_messages
    # 再次回归到 主管节点，进行下一步规划调度
    result = Command(
        goto="supervisor",
        update=update_payload
    )
    logger.info(f"====== supervisor_tools end (going to supervisor) ======")
    return result
