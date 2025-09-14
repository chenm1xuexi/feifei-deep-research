from langgraph.constants import END, START
from langgraph.graph import StateGraph

from deep_research.graph.context import StaticContext
from deep_research.graph.nodes import clarify_with_user, write_research_brief, final_report_generation
from deep_research.graph.state import DeepResearchState, AgentInputState
from deep_research.graph.supervisor.builder import supervisor_subgraph


def build_graph():
    """
    主graph，核心入口，从用户问题澄清 到最终的研究报告生成
    """
    deep_researcher_builder = StateGraph(state_schema=DeepResearchState,
                                         input_schema=AgentInputState,
                                         context_schema=StaticContext)
    deep_researcher_builder.add_node("clarify_with_user", clarify_with_user)  # 用户问题澄清节点
    deep_researcher_builder.add_node("write_research_brief", write_research_brief)  # 深度研究主题任务规划节点
    deep_researcher_builder.add_node("research_supervisor", supervisor_subgraph)  # 深度研究执行节点
    deep_researcher_builder.add_node("final_report_generation", final_report_generation)  # 报告生成节点

    deep_researcher_builder.add_edge(START, "clarify_with_user")
    deep_researcher_builder.add_edge("research_supervisor", "final_report_generation")  # 深度研究完成后，进入到报告生成节点
    deep_researcher_builder.add_edge("final_report_generation", END)  # 报告完成后，即退出

    return deep_researcher_builder.compile()

deep_researcher = build_graph()

