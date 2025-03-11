from typing import Literal

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph
from langgraph.types import Command

from deep_research.nodes.report_nodes import (
    GenerateReportPlanNode,
    HumanFeedbackNode,
    GatherCompletedSectionsNode,
    CompileFinalReportNode,
    InitiateNoResearchSectionsWritingNode,
)
from deep_research.nodes.section_nodes import (
    GenerateQueriesNode,
    SearchWebNode,
    WriteSectionNode,
    SectionStepNode, WriteNoResearchSectionNode,
)
from deep_research.state import (
    ReportStateOutput,
    SectionOutputState,
    ReportState,
    SectionState,
)


async def generate_report_plan(state: ReportState, config: RunnableConfig):
    return await GenerateReportPlanNode().ainvoke(state, config)


async def human_feedback(state: ReportState, config: RunnableConfig) -> Command[
    Literal["generate_report_plan", "build_section_with_web_research"]]:
    return await HumanFeedbackNode().ainvoke(state, config)


async def init_section_step(state: SectionState, config: RunnableConfig):
    return await SectionStepNode().ainvoke(state, config)


async def generate_queries(state: SectionState, config: RunnableConfig):
    return await GenerateQueriesNode().ainvoke(state, config)


async def search_web(state: SectionState, config: RunnableConfig):
    return await SearchWebNode().ainvoke(state, config)


async def write_section(state: SectionState, config: RunnableConfig) -> Command[Literal[END, "search_web"]]:
    return await WriteSectionNode().ainvoke(state, config)


async def gather_completed_sections(state: ReportState, config: RunnableConfig):
    return await GatherCompletedSectionsNode().ainvoke(state, config)


async def write_no_research_section(state: SectionState, config: RunnableConfig):
    return await WriteNoResearchSectionNode().ainvoke(state, config)


async def compile_final_report(state: ReportState, config: RunnableConfig):
    return await CompileFinalReportNode().ainvoke(state, config)


async def initiate_no_research_sections_writing(state: ReportState, config: RunnableConfig):
    return await InitiateNoResearchSectionsWritingNode().ainvoke(state, config)


# 开始构建langgraph 工作流
# 1. 先创建 章节的子工作流
section_builder = StateGraph(input=SectionState, output=SectionOutputState)
section_builder.add_node("init_section_step", init_section_step)
section_builder.add_node("generate_queries", generate_queries)
section_builder.add_node("search_web", search_web)
section_builder.add_node("write_section", write_section)

section_builder.set_entry_point("init_section_step")
section_builder.add_edge("init_section_step", "generate_queries")
section_builder.add_edge("generate_queries", "search_web")
section_builder.add_edge("search_web", "write_section")

# 2. 构建 报告的工作流
report_builder = StateGraph(input=ReportState, output=ReportStateOutput)
report_builder.add_node("generate_report_plan", generate_report_plan)
report_builder.add_node("human_feedback", human_feedback)
# 为当前节点插入章节子流程
report_builder.add_node("build_section_with_web_research", section_builder.compile())
report_builder.add_node("gather_completed_sections", gather_completed_sections)
report_builder.add_node("write_no_research_section", write_no_research_section)
report_builder.add_node("compile_final_report", compile_final_report)

report_builder.set_entry_point("generate_report_plan")
report_builder.add_edge("generate_report_plan", "human_feedback")
report_builder.add_edge("build_section_with_web_research", "gather_completed_sections")
report_builder.add_conditional_edges("gather_completed_sections",
                                     initiate_no_research_sections_writing,
                                     ["write_no_research_section"])
report_builder.add_edge("write_no_research_section", "compile_final_report")
report_builder.add_edge("compile_final_report", END)
