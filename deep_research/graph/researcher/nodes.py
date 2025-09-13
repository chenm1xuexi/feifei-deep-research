from langgraph.runtime import Runtime

from deep_research.graph.context import StaticContext
from deep_research.graph.researcher.state import ResearcherState
from deep_research.graph.supervisor.state import ResearchComplete
from deep_research.graph.tools import think_tool


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
    tools = [ResearchComplete, think_tool, web_search_tool]

