from typing import Literal

import chainlit as cl
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command, Send

from deep_research.config.application_project import REPORT_STRUCTURE, NUMBER_OF_QUERIES
from deep_research.llm.llm import ModelRouter
from deep_research.nodes import BaseNode
from deep_research.prompts import REPORT_PLANNER_QUERY_WRITER_PROMPT, REPORT_PLANNER_PROMPT
from deep_research.state import ReportState, Queries
from deep_research.utils import to_sections, format_sections, now, \
    web_search


class GenerateReportPlanNode(BaseNode):
    """

    生成包含章节的初始报告计划。
    此节点的功能：
    1. 获取报告结构和搜索参数的配置。
    2. 生成搜索查询以收集用于计划的上下文信息。
    3. 使用这些查询执行网络搜索。
    4. 使用 LLM（大语言模型）生成一个包含章节的结构化计划。

    """

    def get_node_name(self):
        return "generate_report_plan"

    async def ainvoke(self, state: ReportState, config: RunnableConfig):
        # 获取状态机 相关属性
        topic = state["topic"]
        feedback = state.get("feedback_on_report_plan", None)

        report_structure = REPORT_STRUCTURE

        writer_model = ModelRouter().get_model()
        # 设置生成联网搜索查询的 system prompt
        generate_query_system_prompt = REPORT_PLANNER_QUERY_WRITER_PROMPT.format(topic=topic,
                                                                                 report_organization=report_structure,
                                                                                 number_of_queries=NUMBER_OF_QUERIES,
                                                                                 now=now())

        # 设置生成联网搜索查询的 user prompt
        generate_query_user_prompt = "生成有助于规划报告章节的搜索查询"

        # 设置模型结构化输出
        query_structured_llm = writer_model.with_structured_output(Queries)

        async with cl.Step(name="生成报告规划联网搜索查询",
                           default_open=True) as query_step:
            query_step.input = topic

            # 调用大模型 用于生成联网搜索查询列表
            results = query_structured_llm.invoke([
                SystemMessage(content=generate_query_system_prompt),
                HumanMessage(content=generate_query_user_prompt)
            ])

            # 进行联网搜索
            query_list = [query.search_query for query in results.queries]
            queries_str = '\n'.join(query for query in query_list)

            print(f"规划报告联网搜索查询：\n{queries_str}")
            query_step.output = f"规划报告联网搜索查询：\n{queries_str}"

        async with cl.Step(name="报告规划联网搜索",
                           default_open=True) as search_step:
            search_step.input = queries_str
            # 使用联网搜索
            source_str = await web_search(query for query in query_list)
            # 将检索结果 返回给前端展示
            search_step.output = f"规划报告联网搜索结果：\n\n{source_str}"

        # 设置规划的 user prompt
        planner_user_prompt = """请生成报告的各章节。您的响应json最外层包含一个“sections”字段，该字段包含一个章节列表。
                                每个章节必须包含：名称（name）、描述(description)、研究(research)和内容(content)字段。
                                """

        # 设置报告规划的system prompt
        planner_system_prompt = REPORT_PLANNER_PROMPT.format(topic=topic,
                                                             report_organization=report_structure,
                                                             context=source_str,
                                                             feedback=feedback,
                                                             now=now(),
                                                             )

        # 初始化 规划大模型
        planner_llm = ModelRouter().get_reasoner_model()

        sections_json_str = ""
        begin = False
        async with cl.Step(name="报告规划深度思考",
                           default_open=True) as deep_step:
            # 这里进行简单的流式输出 展示思维链过程
            prompts = [
                SystemMessage(content=planner_system_prompt),
                HumanMessage(content=planner_user_prompt)
            ]
            for chunk in planner_llm.stream(prompts):
                if chunk.additional_kwargs.get("reasoning_content", ""):
                    await deep_step.stream_token(chunk.additional_kwargs["reasoning_content"])
                else:
                    if not begin and chunk.content:
                        await deep_step.stream_token("\n\n")
                    if chunk.content:
                        begin = True
                        await deep_step.stream_token(chunk.content)
                        sections_json_str += chunk.content

        chain = JsonOutputParser() | to_sections
        report_sections = chain.invoke(sections_json_str)
        # 因为deepseek-r1不支持function calling 需要使用prompt来完善
        # planner_chain = planner_llm | JsonOutputParser() | to_sections
        # structured_planner_llm = planner_llm.with_structured_output(Sections)
        # report_sections = await planner_chain.ainvoke([
        #     SystemMessage(content=planner_system_prompt),
        #     HumanMessage(content=planner_user_prompt)
        # ])

        sections = report_sections.sections

        async with cl.Step(name="生成报告规划大纲") as report_step:
            # 将生成的章节内容进行展示
            sections_str = "\n\n".join(
                f"章节: {section.name}\n"
                f"描述: {section.description}\n"
                f"是否需要进行研究: {'Yes' if section.research else 'No'}\n"
                for section in sections
            )
            await cl.Message(content=f"规划报告大纲：\n{sections_str}").send()

        # 添加到状态机中
        return {"sections": sections}


class HumanFeedbackNode(BaseNode):
    """
        获取关于报告计划的人工反馈并指导下一步行动。
        此节点：
        1.格式化当前的报告计划以供人工审查
        2.通过中断获取反馈
        3.根据反馈指导至以下任一环节：
            - 如果计划获得批准，则进入章节撰写
            - 如果提供了反馈，则重新生成计划

        参数：
        state: 包含待审查部分的当前图状态
        config: 工作流程的配置

        返回：
        重新生成计划或开始章节撰写的指令
    """

    def get_node_name(self) -> str:
        return "human_feedback"

    async def ainvoke(self, state: ReportState, config: RunnableConfig) -> Command[
        Literal["generate_report_plan", "build_section_with_web_research"]]:

        topic = state["topic"]
        sections = state["sections"]

        async with cl.Step(name="用户反馈",
                           default_open=True) as feedback_step:
            # 中断消息 提供给用户进一步审查 然后提供反馈建议
            interrupt_message = f"""请对报告计划提供反馈：
                            \n该报告计划是否符合您的需求？\n如果通过，请输入 'true' 以批准该报告计划。\n或者，提供反馈以重新生成报告计划：
                        """
            resp = await cl.AskUserMessage(content=interrupt_message, timeout=60 * 5).send()
            feedback = resp['output']
            feedback_step.output = f"用户反馈已完成 => {feedback}"

        if not feedback:
            await cl.Message(
                content=f"服务错误了，因为你提供的{feedback}不合法",
            ).send()
            raise TypeError("提供反馈的信息不完整或者类型不被支持！")
        else:
            if (isinstance(feedback, bool) and feedback is True) or (
                    isinstance(feedback, str) and (feedback == 'true' or feedback == 'True')):
                # 如果用户觉得符合预期，则开始进行各章节的撰写工作 （这里是并行调度）
                # 这里有一个问题，当所有章节都不需要进行研究时，这里就直接返回空了
                # 需要进行一次兜底 当都不需要进行研究时，也路由到重新生成报告节点，且要求进行研究
                research_sections = [section for section in sections if section.research]
                if len(research_sections) > 0:
                    return Command(goto=[
                        Send("build_section_with_web_research",
                             {"topic": topic, "section": section, "search_iterations": 0})
                        for section in sections if section.research
                    ])
                else:
                    return Command(goto="generate_report_plan",
                                   update={"feedback_on_report_plan": "请重新生成报告，要求对部分章节进行必要的联网搜索研究"})
            elif isinstance(feedback, str):
                # 说明用户不满意，并提供了修改要求
                return Command(goto="generate_report_plan", update={"feedback_on_report_plan": feedback})
            else:
                # 其他情况，直接报错
                raise TypeError("提供反馈的信息不完整或者类型不被支持！")


class GatherCompletedSectionsNode(BaseNode):
    """
    将已完成的章节格式化为撰写最终章节的上下文。
    此节点将所有已完成的研究章节格式化成一个单一的上下文字符串，用于撰写总结部分。
    """

    def get_node_name(self) -> str:
        return "gather_completed_sections"

    async def ainvoke(self, state: ReportState, config: RunnableConfig):
        completed_sections = state["completed_sections"]
        # 对章节进行格式化 返回一个最终的字符串
        completed_report_sections = format_sections(completed_sections)
        async with cl.Step(name="格式化所有章节内容",
                           default_open=True) as gather_step:
            gather_step.output = completed_report_sections

        return {"report_sections_from_research": completed_report_sections}


class CompileFinalReportNode(BaseNode):
    """
    将所有章节编译成最终报告。
    此节点：
    1. 获取所有已完成的章节
    2. 根据原始计划对其进行排序
    3. 将其合并到最终报告中
    """

    def get_node_name(self) -> str:
        return "compile_final_report"

    async def ainvoke(self, state: ReportState, config: RunnableConfig):
        sections = state["sections"]
        completed_sections = {s.name: s.content for s in state["completed_sections"]}
        for section in sections:
            section.content = completed_sections[section.name]

        all_sections = "\n\n".join([s.content for s in sections])
        final_report = f"最终报告：\n{all_sections}"
        print(final_report)
        async with cl.Step(name="生成最终报告") as final_step:
            await cl.Message(content=final_report).send()

        return {"final_report": all_sections}


class InitiateNoResearchSectionsWritingNode(BaseNode):
    """
    创建用于编写非研究章节的并行任务。
    此边缘函数识别不需要研究的章节并
    为每个章节创建并行写作任务。
    """

    def get_node_name(self) -> str:
        return "initiate_no_research_sections_writing"

    async def ainvoke(self, state: ReportState, config: RunnableConfig):
        return [
            Send("write_no_research_section", {"topic": state["topic"],
                                               "section": section,
                                               "sections_from_research": state["report_sections_from_research"]
                                               })
            for section in state["sections"] if not section.research
        ]
