import asyncio

from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
import requests
import os
from dotenv import load_dotenv
from datetime import datetime

from tavily import AsyncTavilyClient, TavilyClient

from deep_research.state import Section, Sections, Feedback
from deep_research.config.application_project import WEB_SEARCH_TYPE, WEB_SEARCH_MAX_RESULTS

load_dotenv()


async def web_search(search_queries):
    """ 联网搜索通用接口 """
    if WEB_SEARCH_TYPE == 'tavily':
        search_results = await tavily_search(search_queries)
        return deduplicate_and_format_sources(search_results)
    elif WEB_SEARCH_TYPE == 'duckduckgo':
        search_results = await duckduckgo_search(search_queries)
        return deduplicate_and_format_sources(search_results)
    elif WEB_SEARCH_TYPE == 'bocha':
        search_results = await bocha_search(search_queries)
        return deduplicate_and_format_sources(search_results)
    else:
        raise ValueError(f"不支持此联网搜索类型（WEB_SEARCH_TYPE）:{WEB_SEARCH_TYPE}")


async def tavily_search(search_queries):
    """ tavily 联网搜索接口 """

    # 同步调度
    # tavily_client = TavilyClient()
    # search_docs = []
    # for query in search_queries:
    #     search_docs.append(
    #         tavily_client.search(
    #             query,
    #             max_results=WEB_SEARCH_MAX_RESULTS,
    #             include_raw_content=False,
    #             topic="general"
    #         )
    #     )

    # 异步存在 并发上限异常问题
    tavily_async_client = AsyncTavilyClient()
    search_tasks = []
    for query in search_queries:
        search_tasks.append(
            tavily_async_client.search(
                query,
                max_results=5,
                include_raw_content=False,
                topic="general"
            )
        )

    # Execute all searches concurrently
    search_docs = await asyncio.gather(*search_tasks)

    return search_docs


async def duckduckgo_search(search_queries):
    """ duckduckgo 联网搜索接口 """
    wrapper = DuckDuckGoSearchAPIWrapper(region="cn-zh", time="d", source="text", max_results=WEB_SEARCH_MAX_RESULTS)
    search_client = DuckDuckGoSearchResults(api_wrapper=wrapper, output_format="list")
    search_docs = []
    for query in search_queries:
        results = []
        pages = search_client.invoke(query)
        for page in pages:
            results.append({
                "title": page["title"],
                "url": page['link'],
                "content": page['snippet'],
            })
        search_docs.append(
            {
                "query": query,
                "follow_up_questions": None,
                "answer": None,
                "images": [],
                "results": results
            }
        )

    return search_docs


async def bocha_search(search_queries):
    """ 博查联网搜索接口 """
    url = "https://api.bochaai.com/v1/web-search"
    headers = {
        "Authorization": f"Bearer {os.getenv('BOCHA_API_KEY')}",
        "Content-type": "application/json"
    }

    search_docs = []
    for query in search_queries:
        data = {
            "query": query,
            "freshness": "noLimit",
            "summary": True,
            "count": WEB_SEARCH_MAX_RESULTS
        }

        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 200:
            json_resp = response.json()
            try:
                if json_resp["code"] != 200 or not json_resp["data"]:
                    return f"博查网络搜索失败，原因是：{response.msg or '未知错误'}"
                webpages = json_resp["data"]["webPages"]["value"]
                if not webpages:
                    return "未找到相关结果."
                results = []
                for idx, page in enumerate(webpages, start=1):
                    results.append({
                        "title": page["name"],
                        "url": page['url'],
                        "content": page['summary'],
                    })
                search_docs.append({
                    "query": query,
                    "follow_up_questions": None,
                    "answer": None,
                    "images": [],
                    "results": results
                })

                return search_docs
            except Exception as e:
                return f"搜索API请求失败，原因是：搜索结果解析失败 {str(e)}"
        else:
            return f"搜索API请求失败，状态码: {response.status_code}, 错误信息: {response.text}"


def deduplicate_and_format_sources(search_response):
    """
    去重联网搜索结果 以及格式化 来源信息为字符串
    """
    formatted_text = "内容来源:\n"

    if len(search_response) < 1:
        return formatted_text

    sources_list = []
    for response in search_response:
        sources_list.extend(response.get('results', []))

    # 去重链接来源
    unique_sources = {source['url']: source for source in sources_list}

    # 格式化输出
    for i, source in enumerate(unique_sources.values(), 1):
        formatted_text += "\n\n"
        formatted_text += f"标题: {source['title']}\n"
        formatted_text += f"链接地址: {source['url']}\n"
        formatted_text += f"链接内容摘要: {source['content']}\n"
        formatted_text += "\n\n"
    return formatted_text.strip()


def format_sections(sections: list[Section]) -> str:
    """ 将章节内容格式化为字符串 """
    formatted_str = ""
    for idx, section in enumerate(sections, 1):
        formatted_str += f"""
        {'============'}
        章节 {idx} 名称: {section.name}
        {'============'}
        章节主题:
        {section.description}
        是否需要研究: 
        {section.research}
        章节内容:
        {section.content if section.content else '[还没开始写]'}
        """
    return formatted_str


def to_sections(inputs: dict):
    return Sections(**inputs)


def to_feedback(inputs: dict):
    return Feedback(**inputs)


def now():
    # 获取当前日期和时间
    current_datetime = datetime.now()

    # 格式化日期为 yyyy-mm-dd
    return current_datetime.strftime('%Y-%m-%d')
