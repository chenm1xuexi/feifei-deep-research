from typing import Annotated, List, TypedDict, Literal

from pydantic import BaseModel, Field
import operator


class Section(BaseModel):
    """ 章节信息 class """
    name: str = Field(description="当前章节报告的名称")
    description: str = Field(description="当前章节涵盖主题和概念的简要概述")
    research: bool = Field(description="是否需要未当前章节进行联网搜索研究")
    content: str = Field(description="当前章节的内容信息")


class Sections(BaseModel):
    """ 章节列表 """
    sections: List[Section] = Field(description="主题报告的章节列表")


class SearchQuery(BaseModel):
    """ 联网搜索查询 """
    search_query: str = Field(None, description="联网搜索查询")


class Queries(BaseModel):
    """ 联网搜索查询列表 """
    queries: List[SearchQuery] = Field(description="联网搜素查询列表")


class Feedback(BaseModel):
    """ 反馈 """
    grade: Literal["pass", "fail"] = Field(description="评估结果，指示响应是否符合要求（'pass'）或需要修订（'fail'）。")
    follow_up_queries: List[SearchQuery] = Field(description="后续联网搜索查询的列表")


class ReportStateInput(TypedDict):
    """ 主题输入 """
    topic: str


class ReportStateOutput(TypedDict):
    """ 最终报告 """
    final_report: str


class ReportState(TypedDict):
    """ 报告状态机 """
    # 报告主题
    topic: str
    # 报告计划的反馈
    feedback_on_report_plan: str
    # 报告的章节
    sections: list[Section]
    # 已完成的章节列表
    completed_sections: Annotated[list, operator.add]
    # 根据研究完成的章节内容字符串，用来撰写最终章节
    report_sections_from_research: str
    # 最终报告
    final_report: str


class SectionState(TypedDict):
    """ 章节报告状态机 """
    # 报告主题
    topic: str
    # 报告的章节
    section: Section
    # 已完成的联网搜索迭代次数
    search_iterations: int
    # 联网搜索查询列表
    search_queries: list[SearchQuery]
    # 从联网搜索中获取的格式化来源内容的字符串
    source_str: str
    # 根据研究完成的章节内容字符串，用来撰写最终章节
    report_sections_from_research: str
    # 最终章节列表
    completed_sections: list[Section]
    # 当前父节点 step
    parent_step_id: str


class SectionOutputState(TypedDict):
    """ 章节输出状态机 """
    #  已完成的章节
    completed_sections: list[Section]




