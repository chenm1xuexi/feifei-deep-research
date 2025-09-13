from langgraph.graph import StateGraph

from deep_research.graph.context import StaticContext
from deep_research.graph.nodes import clarify_with_user, write_research_brief
from deep_research.graph.state import DeepResearchState, AgentInputState


def build_graph():
    """
    主graph，核心入口，从用户问题澄清 到最终的研究报告生成
    """
    deep_researcher_builder = StateGraph(DeepResearchState,
                                         input_schema=AgentInputState,
                                         context_schema=StaticContext)
    deep_researcher_builder.add_node("clarify_with_user", clarify_with_user)  # 用户问题澄清节点
    deep_researcher_builder.add_node("write_research_brief", write_research_brief)  # 深度研究主题任务规划节点
    deep_researcher_builder.add_node("research_supervisor", supervisor_subgraph)  # 深度研究执行节点
    deep_researcher_builder.add_node("final_report_generation", final_report_generation)  # 报告生成节点
