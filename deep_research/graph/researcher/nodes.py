import asyncio
from typing import Literal

from langchain_core.messages import SystemMessage, ToolMessage, HumanMessage
from langchain_core.messages.utils import (
    trim_messages,
    count_tokens_approximately, filter_messages
)
from langgraph.runtime import Runtime
from langgraph.types import Command

from deep_research.graph.context import StaticContext
from deep_research.graph.researcher.prompts import research_system_prompt, compress_research_simple_human_message, \
    compress_research_system_prompt
from deep_research.graph.researcher.state import ResearcherState
from deep_research.graph.tools import execute_tool_safely, get_all_tools
from deep_research.llms.llm import get_llm
from deep_research.utils.utils import now


async def researcher(state: ResearcherState, runtime: Runtime[StaticContext]):
    """
        进行特定主题深度研究的独立研究员。

        该研究员由主管分配具体的研究主题，使用可用工具（搜索、思考工具、MCP工具）
        来收集全面的信息。它可以在搜索之间使用思考工具进行战略规划和反思

        参数:
            state: 包含消息和主题上下文的当前研究员状态
            config: 包含模型设置和工具可用性的运行时配置

        返回:
            命令以继续到researcher_tools执行工具
        """

    static_context = runtime.context
    researcher_messages = state.get("researcher_messages", [])

    # 获取研究的所有工具，包含mcp
    tools = await get_all_tools(runtime)

    # 定义研究员的提示词
    researcher_prompt = research_system_prompt.format(
        date_time=now(),
    )

    # 定义研究模型 + 结构化输出
    research_model = get_llm(
        model=static_context.research_model,
        max_tokens=static_context.research_model_max_tokens  # 定义研究结果的最大输出token
    ).bind_tools(tools)

    # 生成研究成果
    messages = [SystemMessage(content=researcher_prompt)] + researcher_messages
    response = await research_model.ainvoke(messages)

    # 更新状态 + 执行工具调度
    return Command(
        goto="researcher_tools",
        update={
            "researcher_messages": [response],
            "tool_call_iterations": state.get("tool_call_iterations", 0) + 1,
        }
    )


async def researcher_tools(
        state: ResearcherState,
        runtime: Runtime[StaticContext],
) -> Command[Literal["researcher", "compress_research"]]:
    """
    执行研究员调用的工具，包括搜索工具和战略思考。

        该函数处理各种类型的研究员工具调用：
        1. think_tool - 战略反思，继续研究对话
        2. 搜索工具 (web_search_tool) - 信息收集
        3. MCP 工具 - 外部工具集成
        4. ResearchComplete - 表示单个研究任务完成的信号

        参数:
            state: 包含消息和迭代次数的当前研究员状态
            runtime: 静态运行上下文
        返回:
            命令以继续研究循环或进入压缩阶段
        """

    static_context = runtime.context
    researcher_messages = state.get("researcher_messages", [])
    # 获取最近的ai message 提取工具调用
    most_recent_message = researcher_messages[-1]
    has_tool_calls = bool(most_recent_message.tool_calls)
    if not has_tool_calls:
        # 不存在工具调度时，转交给 压缩阶段
        return Command(goto="compress_research")

    tool_calls = most_recent_message.tool_calls

    tools = await get_all_tools(runtime)
    tools_by_name = {
        tool.name if hasattr(tool, "name") else tool.get("name", "web_search"): tool
        for tool in tools
    }

    tool_execution_tasks = [
        execute_tool_safely(tools_by_name[tool_call["name"]], tool_call["args"], runtime)
        for tool_call in tool_calls
    ]

    # 并行调度，获取工具调用结果
    observations = await asyncio.gather(*tool_execution_tasks)

    tool_outputs = [
        ToolMessage(
            content=observation,
            name=tool_call["name"],
            tool_call_id=tool_call["id"]
        )
        for observation, tool_call in zip(observations, tool_calls)
    ]

    # 工具调用完成后，检查是否达到调用工具最大次数上限
    exceeded_iterations = state.get("tool_call_iterations", 0) >= static_context.max_react_tool_calls
    research_complete_called = any(
        tool_call["name"] == "ResearchComplete"
        for tool_call in most_recent_message.tool_calls
    )

    # 如果超过最大调用次数或者调用了ResearchComplete，则转交给 压缩阶段
    if exceeded_iterations or research_complete_called:
        return Command(
            goto="compress_research",
            update={"researcher_messages": tool_outputs}
        )

    # 其他情况，继续进行研究
    return Command(
        goto="researcher",
        update={"researcher_messages": tool_outputs}
    )

async def compress_research(
        state: ResearcherState,
        runtime: Runtime[StaticContext],
):
    """将研究发现压缩并综合成一个简洁、结构化的摘要。

    该函数获取研究员工作中的所有研究发现、工具输出和AI消息，并将其提炼成一个
    干净、全面的摘要，同时保留所有重要信息和发现。

    参数:
        state: 包含累积研究消息的当前研究员状态
        config: 包含压缩模型设置的运行时配置

    返回:
        包含压缩研究摘要和原始笔记的字典
    """

    static_context = runtime.context
    researcher_messages = state.get("researcher_messages", [])
    # 添加压缩提示词
    researcher_messages.append(HumanMessage(content=compress_research_simple_human_message))

    # 这里需要注意上下文上限问题，暂时采用截断

    compression_prompt = compress_research_system_prompt.format(date_time=now())
    messages = [SystemMessage(content=compression_prompt)] + researcher_messages

    # 在调用前，对当前消息上下文 进行校验 是否超过上下文限制，超过则进行trim
    messages = trim_messages(
        messages=messages,
        strategy="last",
        max_tokens=static_context.compression_model_context_length,
        token_counter=count_tokens_approximately,
        include_system=True,
        end_on=("human", "tool"),  # 保留最后的human 和 tool的信息，忽略ai
    )

    # 裁剪完成后，完成当前主题的研究压缩工作
    synthesizer_model = get_llm(
        model=static_context.compression_model,
        max_tokens=static_context.compression_model_max_tokens
    )

    response = await synthesizer_model.ainvoke(messages)

    # 从所有工具和AI消息中提取ai信息 和 工具结果信息
    raw_notes_content = "\n".join([
        str(message.content)
        for message in filter_messages(researcher_messages, include_types=["tool", "ai"])
    ])

    return {
        "compressed_research": str(response.content),
        "raw_notes": [raw_notes_content]
    }