from langgraph.constants import END
from langgraph.graph import StateGraph

from deep_research.graph.context import StaticContext
from deep_research.graph.researcher.nodes import researcher, researcher_tools, compress_research
from deep_research.graph.researcher.state import ResearcherState, ResearcherOutputState


def build_researcher_graph():
    """ 构建研究员 """
    researcher_builder = StateGraph(
        state_schema=ResearcherState,
        output_schema=ResearcherOutputState,
        context_schema=StaticContext,
    )

    researcher_builder.add_node("researcher", researcher)  # 核心研究节点
    researcher_builder.add_node("researcher_tools", researcher_tools)  # 工具执行节点
    researcher_builder.add_node("compress_research", compress_research)  # 研究成果总结压缩节点

    researcher_builder.set_entry_point("researcher")
    researcher_builder.add_edge("compress_research", END)

    return researcher_builder.compile()


researcher_subgraph = build_researcher_graph()
