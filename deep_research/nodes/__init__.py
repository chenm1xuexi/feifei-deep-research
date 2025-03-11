from abc import ABC, abstractmethod

from langchain_core.runnables import RunnableConfig

from deep_research.state import ReportState, SectionState


class BaseNode(ABC):
    """ 报告流程节点 基类 """

    @abstractmethod
    def get_node_name(self) -> str:
        """ 获取节点名称 """
        raise NotImplementedError("未实现")

    @abstractmethod
    async def ainvoke(self, state: ReportState, config: RunnableConfig):
        """ 异步调度 """
        raise NotImplementedError("未实现")


class BaseSectionNode(ABC):
    """ 章节流程节点 基类 """

    @abstractmethod
    def get_node_name(self) -> str:
        """ 获取节点名称 """
        raise NotImplementedError("未实现")

    @abstractmethod
    async def ainvoke(self, state: SectionState, config: RunnableConfig):
        """ 异步调度 """
        raise NotImplementedError("未实现")
