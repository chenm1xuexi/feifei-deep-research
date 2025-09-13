import asyncio
from typing import List, Annotated, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import MessageLikeRepresentation, filter_messages, HumanMessage
from langchain_core.tools import tool, InjectedToolArg
from pydantic import BaseModel, Field
from tavily import AsyncTavilyClient

from langgraph.runtime import get_runtime

from common.log import logger
from deep_research.graph.context import StaticContext
import os

from dotenv import load_dotenv

from deep_research.graph.researcher.prompts import summarize_webpage_prompt
from deep_research.graph.researcher.state import Summary
from deep_research.llms.llm import get_llm
from deep_research.utils.utils import now

load_dotenv()


class ThinkToolArgsSchema(BaseModel):
    """ 反思工具 """
    reflection: str = Field(description="Your detailed reflection on research progress, findings, gaps, and next steps")


@tool(description="""
Tool for strategic reflection on research progress and decision-making.

    Use this tool after each search to analyze results and plan next steps systematically.
    This creates a deliberate pause in the research workflow for quality decision-making.

    When to use:
    - After receiving search results: What key information did I find?
    - Before deciding next steps: Do I have enough to answer comprehensively?
    - When assessing research gaps: What specific information am I still missing?
    - Before concluding research: Can I provide a complete answer now?

    Reflection should address:
    1. Analysis of current findings - What concrete information have I gathered?
    2. Gap assessment - What crucial information is still missing?
    3. Quality evaluation - Do I have sufficient evidence/examples for a good answer?
    4. Strategic decision - Should I continue searching or provide my answer?
""",
      args_schema=ThinkToolArgsSchema)
def think_tool(reflection: str):
    return f"Reflection recorded: {reflection}"


@tool(description="""
A search engine optimized for comprehensive, accurate, and trusted results. 
Useful for when you need to answer questions about current events.
""")
async def web_search_tool(
        queries: List[str],
        max_results: Annotated[int, InjectedToolArg] = 5,
        topic: Annotated[Literal["general", "news", "finance"], InjectedToolArg] = "general",
) -> str:
    """从 Tavily 搜索 API 获取并总结搜索结果。

        参数：
            queries: 要执行的搜索查询列表
            max_results: 每个查询返回的最大结果数
            topic: 搜索结果的主题过滤器（通用、新闻或金融）
            config: API 密钥和模型设置的运行时配置

        返回：
            包含总结搜索结果的格式化字符串
        """
    # 1. 调度搜索引擎获取搜索结果
    search_results = await tavily_search_async(
        queries,
        max_results=max_results,
        topic=topic,
        include_raw_content=True
    )

    # 2. 去重搜索link
    unique_results = {}
    for response in search_results:
        for result in response['results']:
            url = result['url']
            if url not in unique_results:
                unique_results[url] = {**result, "query": response['query']}

    # 3. 定义总结摘要的模型实例 和结构化输出
    runtime = get_runtime(StaticContext)
    static_context = runtime.context

    # 获取允许网页
    max_char_to_include = static_context.max_content_length

    # 获取对网页内容进行总结的模型
    summarization_model = get_llm(
        model=static_context.summarization_model,
        max_tokens=static_context.summarization_model_max_tokens,
    ).with_structured_output(Summary)

    # 4. 创建总结摘要的任务
    async def noop():
        """No-op function for results without raw content."""
        return None

    summarization_tasks = [
        noop() if not result.get("raw_content")
        else summarize_webpage(
            summarization_model,
            result['raw_content'][:max_char_to_include]
        )
        for result in unique_results.values()
    ]

    # Step 5: Execute all summarization tasks in parallel
    summaries = await asyncio.gather(*summarization_tasks)

    # Step 6: Combine results with their summaries
    summarized_results = {
        url: {
            'title': result['title'],
            'content': result['content'] if summary is None else summary
        }
        for url, result, summary in zip(
            unique_results.keys(),
            unique_results.values(),
            summaries
        )
    }

    # Step 7: Format the final output
    if not summarized_results:
        return "No valid search results found. Please try different search queries or use a different search API."

    formatted_output = "Search results: \n\n"
    for i, (url, result) in enumerate(summarized_results.items()):
        formatted_output += f"\n\n--- SOURCE {i + 1}: {result['title']} ---\n"
        formatted_output += f"URL: {url}\n\n"
        formatted_output += f"SUMMARY:\n{result['content']}\n\n"
        formatted_output += "\n\n" + "-" * 80 + "\n"

    return formatted_output


async def tavily_search_async(
        search_queries,
        max_results: int = 5,
        topic: Literal["general", "news", "finance"] = "general",
        include_raw_content: bool = True,
):
    """异步执行多个Tavily搜索查询。

    Args:
        search_queries: 要执行的搜索查询字符串列表
        max_results: 每个查询的最大结果数
        topic: 用于过滤结果的主题类别
        include_raw_content: 是否包含完整的网页内容

    Returns:
        来自Tavily API的搜索结果字典列表
    """
    # Initialize the Tavily client with API key from config

    tavily_client = AsyncTavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

    # Create search tasks for parallel execution
    search_tasks = [
        tavily_client.search(
            query,
            max_results=max_results,
            include_raw_content=include_raw_content,
            topic=topic
        )
        for query in search_queries
    ]

    # Execute all search queries in parallel and return results
    search_results = await asyncio.gather(*search_tasks)
    return search_results


def get_notes_from_tool_calls(messages: list[MessageLikeRepresentation]):
    """
    从深度研究的消息上下文中 提取工具调用message信息
    """
    return [tool_msg.content for tool_msg in filter_messages(messages, include_types="tool")]


async def summarize_webpage(model: BaseChatModel, webpage_content: str) -> str:
    """使用AI模型对网页内容进行摘要，并提供超时保护。

    Args:
        model: 配置用于摘要的聊天模型
        webpage_content: 需要被摘要的原始网页内容

    Returns:
        格式化的摘要和关键摘录，如果摘要失败则返回原始内容
    """
    try:
        # 设置总结网页内容的提示词
        prompt_content = summarize_webpage_prompt.format(
            webpage_content=webpage_content,
            date_time=now(),
        )

        # Execute summarization with timeout to prevent hanging
        summary = await asyncio.wait_for(
            model.ainvoke([HumanMessage(content=prompt_content)]),
            timeout=60.0  # 60 second timeout for summarization
        )

        # Format the summary with structured sections
        formatted_summary = (
            f"<summary>\n{summary.summary}\n</summary>\n\n"
            f"<key_excerpts>\n{summary.key_excerpts}\n</key_excerpts>"
        )

        return formatted_summary

    except asyncio.TimeoutError:
        # Timeout during summarization - return original content
        logger.warning("Summarization timed out after 60 seconds, returning original content")
        return webpage_content
    except Exception as e:
        # Other errors during summarization - log and return original content
        logger.warning(f"Summarization failed with error: {str(e)}, returning original content")
        return webpage_content
