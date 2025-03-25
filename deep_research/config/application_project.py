import os
from dotenv import load_dotenv

load_dotenv()

DEFAULT_REPORT_STRUCTURE_TEMPLATE = """
使用此结构创建关于用户提供主题的报告：

1. 简介（无需研究）
- 主题领域的简要概述

2. 主体部分：
- 每个部分应重点关注用户提供主题的子主题

3. 结论
- 目标是 1 个结构元素（列表或表格）来提炼主体部分
- 提供报告的简明摘要  
"""

# 默认的最终生成报告的模版结构
REPORT_STRUCTURE = DEFAULT_REPORT_STRUCTURE_TEMPLATE

# 每次迭代生成的网络检索问题查询的个数
NUMBER_OF_QUERIES = 2

# 反思 + 联网搜索的最大深度
MAX_SEARCH_DEPTH = 1

# 规划者模型
DEEPSEEK_PLANNER_MODEL = {
    "model-name": os.getenv("DEEPSEEK_REASONER_MODEL", "deepseek-reasoner"),
    "api-key": os.getenv("DEEPSEEK_API_KEY"),
    "base-url": os.getenv("DEEPSEEK_BASE_URL"),
}

# 撰写模型
DEEPSEEK_WRITER_MODEL = {
    "model-name": os.getenv("DEEPSEEK_CHAT_MODEL", "deepseek-chat"),
    "api-key": os.getenv("DEEPSEEK_API_KEY"),
    "base-url": os.getenv("DEEPSEEK_BASE_URL"),
}


# 规划者模型
TONGYI_PLANNER_MODEL = {
    "model-name": os.getenv("TONGYI_QWQ_MODEL", "qwq-32b"),
    "api-key": os.getenv("TONGYI_API_KEY"),
    "base-url": os.getenv("TONGYI_BASE_URL"),
}

# 撰写模型
TONGYI_WRITER_MODEL = {
    "model-name": os.getenv("TONGYI_QWEN_MODEL", "qwen-max"),
    "api-key": os.getenv("TONGYI_API_KEY"),
    "base-url": os.getenv("TONGYI_BASE_URL"),
}

# 联网搜索服务 目前仅支持
# WEB_SEARCH_TYPE = "bocha"
WEB_SEARCH_TYPE = "tavily"
# 单次最多获取搜索网页个数
WEB_SEARCH_MAX_RESULTS = 5


# 这里统一采用 通义 相关模型测试 目前仅提供 deepseek 和 tongyi 俩种选择，需要其他的 请自己去拓展
MODEL_PROVIDER = "tongyi"
