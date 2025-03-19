from typing import Literal

import chainlit as cl
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnableConfig
from langgraph.constants import END
from langgraph.types import Command

from deep_research.config.application_project import NUMBER_OF_QUERIES, MAX_SEARCH_DEPTH
from deep_research.llm.llm import ModelRouter
from deep_research.nodes import BaseSectionNode
from deep_research.prompts import QUERY_WRITER_PROMPT, SECTION_WRITER_INPUTS, SECTION_WRITER_USER_PROMPT, \
    SECTION_GRADER_PROMPT, FINAL_SECTION_WRITER_PROMPT
from deep_research.state import SectionState, Queries, NoResearchSectionState
from deep_research.utils import web_search, to_feedback, now


class SectionStepNode(BaseSectionNode):
    """ 主要是为了结合chainlit 来生成 章节的step 根节点 """

    def get_node_name(self) -> str:
        pass

    async def ainvoke(self, state: SectionState, config: RunnableConfig):
        section = state["section"]
        async with cl.Step(name=f"章节 [{section.name}] 深度研究", default_open=True) as section_step:
            pass
        return {"parent_step_id": section_step.id}


class GenerateQueriesNode(BaseSectionNode):
    """
    生成用于研究特定章节的搜索查询。
    此节点使用大型语言模型（LLM）基于章节的主题和描述生成有针对性的搜索查询。
    """

    def get_node_name(self) -> str:
        return "generate_queries"

    async def ainvoke(self, state: SectionState, config: RunnableConfig):
        topic = state["topic"]
        section = state["section"]
        parent_step_id = state["parent_step_id"]

        generate_query_llm = ModelRouter().get_model()

        structured_generate_query_llm = generate_query_llm.with_structured_output(Queries)

        # 设置生成当前章节的联网搜索查询的 system prompt
        generate_section_query_system_prompt = QUERY_WRITER_PROMPT.format(topic=topic,
                                                                          section_topic=section.description,
                                                                          number_of_queries=NUMBER_OF_QUERIES,
                                                                          now=now(),
                                                                          )
        # 设置生成当前章节的联网搜索查询的 user prompt
        generate_section_query_user_prompt = "请生成关于所提供主题的联网搜索查询。"

        prompts = [
            SystemMessage(content=generate_section_query_system_prompt),
            HumanMessage(content=generate_section_query_user_prompt)
        ]

        # 调用大模型生成查询
        queries = structured_generate_query_llm.invoke(prompts)
        query_str = "\n\n".join(query.search_query for query in queries.queries)
        print(f"获取章节[{section.name}]检索查询：\n{query_str}")
        async with cl.Step(name=f"章节 [{section.name}] 生成联网搜索查询",
                           parent_id=parent_step_id) as generate_query_step:
            generate_query_step.output = query_str

        return {"search_queries": queries.queries}


class SearchWebNode(BaseSectionNode):
    """
        执行针对当前章节查询的联网搜索。
        此节点：
        1. 获取生成的查询
        2. 使用配置的搜索 API 执行搜索
        3. 将结果格式化为可用的上下文
        """

    def get_node_name(self) -> str:
        return "search_web"

    async def ainvoke(self, state: SectionState, config: RunnableConfig):
        section = state["section"]
        search_queries = state["search_queries"]
        parent_step_id = state["parent_step_id"]
        search_iterations = state["search_iterations"]

        # 使用联网搜索
        source_str = await web_search([query.search_query for query in search_queries])

        async with cl.Step(name=f"章节 [{section.name}] 联网搜索查询结果",
                           parent_id=parent_step_id) as search_web_step:
            print(f"章节 [{section.name}] 联网搜索查询结果: \n{source_str}")
            search_web_step.output = source_str

        return {"source_str": source_str, "search_iterations": search_iterations + 1}


class WriteSectionNode(BaseSectionNode):
    """
        撰写报告的一个章节并评估是否需要进一步研究。
        此节点：
        1. 使用搜索结果撰写章节内容
        2. 评估该章节的质量
        3. 根据评估结果：
            - 如果质量达标，则完成该章节
            - 如果质量不达标，则触发进一步研究
        """

    def get_node_name(self) -> str:
        return "write_section"

    async def ainvoke(self, state: SectionState, config: RunnableConfig) -> Command[Literal[END, "search_web"]]:
        topic = state["topic"]
        section = state["section"]
        source_str = state["source_str"]
        search_iterations = state["search_iterations"]
        parent_step_id = state["parent_step_id"]

        # 报告章节写作 system prompt
        section_writer_system_prompt = SECTION_WRITER_INPUTS.format(topic=topic,
                                                                    section_name=section.name,
                                                                    section_topic=section.description,
                                                                    context=source_str,
                                                                    section_content=section.content)

        # llm生成章节内容
        section_writer_llm = ModelRouter().get_model()

        prompts = [
            SystemMessage(content=section_writer_system_prompt),
            HumanMessage(content=SECTION_WRITER_USER_PROMPT.format(now=now()))
        ]

        section_content_resp_str = ""

        async with cl.Step(name=f"生成章节: [{section.name}] 内容",
                           parent_id=parent_step_id,
                           default_open=True) as section_step:
            for chunk in section_writer_llm.stream(prompts):
                if chunk.content:
                    section_content_resp_str += chunk.content
                    await section_step.stream_token(chunk.content)

        # 获取llm针对当前章节生成的内容 赋值给当前章节对象
        section.content = section_content_resp_str

        # 评估专家对当前章节的内容进行审查
        section_grader_user_message = """
                对报告进行评分，并考虑针对缺失信息的后续问题。
                如果评分为'pass'，则为所有后续查询返回空字符串。
                如果评分为'fail'，则提供具体的搜索查询以收集缺失信息。
            """

        section_grader_system_message = SECTION_GRADER_PROMPT.format(topic=topic,
                                                                     section_topic=section.description,
                                                                     section=section.content,
                                                                     number_of_follow_up_queries=NUMBER_OF_QUERIES,
                                                                     now=now(), )

        # 这里就很重要了，这里需要使用深度思考模型来进行反思 所以我们用deepseek-r1来进行反思
        reflection_llm = ModelRouter().get_reasoner_model()
        prompts = [
            SystemMessage(content=section_grader_system_message),
            HumanMessage(content=section_grader_user_message)
        ]

        is_answering = False
        reflection_content = ""

        async with cl.Step(name=f"章节: [{section.name}] 章节评估 深度思考",
                           parent_id=parent_step_id,
                           default_open=True) as grade_section_step:
            # 深度思考模型开始反思
            for chunk in reflection_llm.stream(prompts):
                if chunk.additional_kwargs.get("reasoning_content", ""):
                    # 返回深度思考流式内容
                    await grade_section_step.stream_token(chunk.additional_kwargs["reasoning_content"])
                else:
                    if is_answering is False and chunk.content != '':
                        is_answering = True
                        await grade_section_step.stream_token("\n\n评估结果内容\n\n")

                    reflection_content += chunk.content
                    await grade_section_step.stream_token(chunk.content)

        feedback_chain = JsonOutputParser() | to_feedback
        feedback = feedback_chain.invoke(reflection_content)

        if feedback.grade == "pass" or search_iterations >= MAX_SEARCH_DEPTH:
            # 如果评估结果通过 或者 超过了检索的最大深度 则对当前章节的撰写直接退出
            async with cl.Step(name=f"章节: [{section.name}] 章节评估",
                               parent_id=parent_step_id) as grade_pass_step:
                grade_pass_step.output = f"当前检索迭代深度：{search_iterations}, 评估结果：通过"
            return Command(
                # 更新当前的状态机
                update={"completed_sections": [section]},
                goto=END
            )
        else:
            # 如果评估结果未通过，则根据提供的新的检索查询 路由到检索节点 重新检索 来补充缺失的主题内容
            feedback_up_queries_str = "\n".join(feedback_up_query.search_query
                                                for feedback_up_query in feedback.follow_up_queries)
            async with cl.Step(name=f"章节: [{section.name}] 章节评估",
                               parent_id=parent_step_id) as grade_fail_step:
                grade_fail_step.output = f"评估结果：未通过，当前检索迭代深度：{state['search_iterations']}, 重新生成的联网搜索查询：\n{feedback_up_queries_str}"

            return Command(
                # 更新当前的状态机
                update={
                    "search_queries": feedback.follow_up_queries,
                    "section": section
                },
                goto="search_web"
            )


class WriteNoResearchSectionNode(BaseSectionNode):
    """
    撰写不需要研究的章节，也就是 research 为 false的章节
    通过使用已完成的章节作为上下文。
    此节点处理诸如结论或摘要，简介等部分，这些部分基于已研究的章节构建，而不需要直接研究。
    简单来说，这个章节 就是总结的章节，不再需要进行联网搜索了 只需要针对前面完成的章节 进行整体的归纳总结
    """

    def get_node_name(self) -> str:
        return "write_no_research_section"

    async def ainvoke(self, state: NoResearchSectionState, config: RunnableConfig):
        topic = state["topic"]
        section = state["section"]
        completed_report_sections = state["sections_from_research"]

        # 设置撰写总结 这一章节 的 system prompt
        no_research_section_writer_system_prompt = FINAL_SECTION_WRITER_PROMPT.format(topic=topic,
                                                                                      section_name=section.name,
                                                                                      section_topic=section.description,
                                                                                      context=completed_report_sections,
                                                                                      now=now(), )

        no_research_section_writer_user_prompt = f"请根据已经提供的资料生成{section.name}报告部分。"

        final_writer_llm = ModelRouter().get_model()
        prompts = [
            SystemMessage(content=no_research_section_writer_system_prompt),
            HumanMessage(content=no_research_section_writer_user_prompt)
        ]

        no_research_section_content = ""
        async with cl.Step(name=f"生成不需要研究的章节 [{section.name}] 内容",
                           default_open=True) as section_no_research_step:
            for chunk in final_writer_llm.stream(prompts):
                if chunk.content:
                    await section_no_research_step.stream_token(chunk.content)
                    no_research_section_content += chunk.content

        section.content = no_research_section_content

        return {"completed_sections": [section]}
