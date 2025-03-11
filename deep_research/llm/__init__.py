from abc import abstractmethod, ABC


class BaseModel(ABC):

    @abstractmethod
    def get_reasoner_model(self):
        """ 获取深度思考模型 """
        raise NotImplementedError()

    @abstractmethod
    def get_model(self):
        """ 获取大语言模型 """
        raise NotImplementedError()
