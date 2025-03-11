from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_community.chat_models import ChatTongyi
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

import os
from dotenv import load_dotenv

from deep_research.state import Sections

load_dotenv()

prompt = """
我需要一个简洁且聚焦的报告计划。
## 任务
你的任务是生成报的各个部分的列表，你的计划应当简洁切集中，避免重复的部分或者不必要的填充内容。
例如，一个好的报告结构应当如下：
1. 引言
2. 主题A概述
3. 主题B概述
4. A与B的比较
5. 结论
   每个部分应当包含以下字段：
    - name(字段类型：字符串) => 该部分的名称。
    - description(字段类型：字符串) => 简要概述该部分的主要内容。
    - research(字段类型：布尔 true/false) => 是否需要对这一部分进行联网搜索研究。
    - content(字段类型：字符串) => 该部分的内容，目前留空。

## 输出结果格式示例
```json
{{
    "sections": [
        {{
            "name": "<当前章节报告的名称>",
            "description": "<当前章节涵盖主题和概念的简要概述>",
            "research": "<是否需要未当前章节进行联网搜索研究, true/false>",
            "content": "<当前章节的内容信息>"
        }},
        {{
            "name": "<当前章节报告的名称>",
            "description": "<当前章节涵盖主题和概念的简要概述>",
            "research": "<是否需要未当前章节进行联网搜索研究, true/false>",
            "content": "<当前章节的内容信息>"
        }},
        {{
            "name": "<当前章节报告的名称>",
            "description": "<当前章节涵盖主题和概念的简要概述>",
            "research": "<是否需要未当前章节进行联网搜索研究, true/false>",
            "content": "<当前章节的内容信息>"
        }}
        
    ]
}}
```

## 注意事项
    - 在主要主题部分内包含示例和实施细节，而不是单独列出。
    - 确保每个部分都有明确的目的，且没有内容重叠。
    - 合并相关概念，而不是分开处理。 
    - 在提交前，审查你的结构，确保没有冗余部分，并且逻辑流畅。 
     - 输出结果为json，而不添加任何额外的解释说明。
"""


def to_sections(inputs: dict):
    return Sections(**inputs)


planner_llm = init_chat_model(model="deepseek-reasoner",
                              model_provider="deepseek")

# planner_llm = ChatTongyi(model="qwq-32b")

prompt = ChatPromptTemplate.from_messages([
    ("system", prompt),
    ("user", "{topic}")
])
chain = prompt | planner_llm
reasoning_content = ""
content = ""
for chunk in chain.stream({"topic": "主题是'就目前流行的深度思考模型进行性能比较'"}):
    if chunk.additional_kwargs.get("reasoning_content", ""):
        reasoning_content += chunk.additional_kwargs["reasoning_content"]
        print(chunk.additional_kwargs["reasoning_content"])
    else:
        content += chunk.content
        print(chunk.content)
