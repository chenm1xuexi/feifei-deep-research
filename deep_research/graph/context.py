from enum import Enum

from pydantic import BaseModel, Field


class SearchAPI(str, Enum):
    """ 联网搜索API."""
    TAVILY = "tavily"
    DUCKDUCKGO = "duckduckgo"
    BOCHA = "bocha"
    NONE = "none"


class StaticContext(BaseModel):
    """
    深度研究 静态运行上下文
    """
    max_structured_output_retries: int = Field(default=3, description="最大结构化输出重试次数")
    allow_clarification: bool = Field(default=True,
                                      description="是否允许 researcher 在开始研究之前询问用户澄清问题, 也就是明确研究方向")

    # 要注意api-key 速率限制，如果开启此配置，最好配置多个api-key 轮询调度
    max_concurrent_research_units: int = Field(default=3,
                                               description="同时运行的最大研究单元数量。这将使 researcher 可以同时使用多个子代理进行研究。")
    search_api: SearchAPI = Field(default=SearchAPI.TAVILY, description="搜索API")
    max_researcher_iterations: int = Field(default=5,
                                           description="researcher 最大研究迭代次数。researcher会反思研究并提出后续问题的次数。")
    max_react_tool_calls: int = Field(default=10, description="research agent 最大可调用工具次数")

    summarization_model: str = Field(default="bigmodel/glm-4.5", description="针对联网搜索结果进行总结的模型选择")
    summarization_model_max_tokens: str = Field(default=100000, description="模型最大输出token")
    max_content_length: int = Field(default=30000,
                                    description="允许网页内容最大长度限制，网页内容将交与总结模型进行生成摘要")

    research_model: str = Field(default="bigmodel/glm-4.5", description="进行深度研究的模型选择")
    research_model_max_context_length: str = Field(default=100000, description="深度研究模型最终输出的最大长度token限制")
    research_model_max_tokens: str = Field(default=50000, description="深度研究模型最终输出的最大长度token限制")

    compression_model: str = Field(default="bigmodel/glm-4.5",
                                   description="对研究智能体的发现成果 整体进行压缩的模型")
    compression_model_context_length: int = Field(default=100000,
                                                  description="上下文压缩模型最大输入token")
    compression_model_max_tokens: str = Field(default=8192, description="上下文压缩模型最大输出token")


    final_report_model: str = Field(default="bigmodel/glm-4.5", description="撰写研究报告的模型选择")
    final_report_model_max_tokens: str = Field(default=20000, description="撰写报告模型最大输出token")

    mcp_config: dict = Field(default={}, description="mcp配置")

    class Config:
        """ 定义允许识别自定义类型"""

        arbitrary_types_allowed = True