from langgraph.graph import StateGraph

from deep_research.graph.context import StaticContext
from deep_research.graph.supervisor.state import SupervisorState


def build_supervisor_graph():

    supervisor_builder = StateGraph(
        state_schema=SupervisorState,
        config_shema=StaticContext,
    )

    supervisor_builder.add_node("supervisor", supervisor)
    supervisor_builder.add_node("supervisor_tools", supervisor_tools)
    supervisor_builder.set_entry_point("supervisor")

    # 返回主管智能体 graph图
    return supervisor_builder.compile()
