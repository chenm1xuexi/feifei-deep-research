from langchain_community.chat_models import ChatTongyi

from deep_research.config.application_project import MODEL_PROVIDER, TONGYI_PLANNER_MODEL, TONGYI_WRITER_MODEL, \
    DEEPSEEK_PLANNER_MODEL, DEEPSEEK_WRITER_MODEL
from deep_research.llm import BaseModel
from langchain_deepseek import ChatDeepSeek


class ModelRouter(BaseModel):
    """ 模型路由器 """

    def get_reasoner_model(self):
        if MODEL_PROVIDER == 'tongyi':
            return TongyiModel().get_reasoner_model()
        elif MODEL_PROVIDER == 'deepseek':
            return DeepSeekModel().get_reasoner_model()
        else:
            raise ValueError(f"不存在此模型服务提供商(MODEL_PROVIDER): {MODEL_PROVIDER}，请检查")

    def get_model(self):
        if MODEL_PROVIDER == 'tongyi':
            return TongyiModel().get_model()
        elif MODEL_PROVIDER == 'deepseek':
            return DeepSeekModel().get_model()
        else:
            raise ValueError(f"不存在此模型服务提供商(MODEL_PROVIDER): {MODEL_PROVIDER}，请检查")


class TongyiModel(BaseModel):
    """ 通义模型 """

    def get_reasoner_model(self):
        # 因为ChatTongyi 还没有适配 思维链，因此这里使用ChatDeepSeek来替代
        return ChatDeepSeek(model=TONGYI_PLANNER_MODEL.get("model-name"),
                            api_key=TONGYI_PLANNER_MODEL.get("api-key"),
                            api_base=TONGYI_PLANNER_MODEL.get("base-url"),
                            temperature=0)

    async def stream(self, inputs):
        yield self.get_reasoner_model().stream(inputs)

    def get_model(self):
        return ChatTongyi(model=TONGYI_WRITER_MODEL.get("model-name"),
                          api_key=TONGYI_WRITER_MODEL.get("api-key"),
                          temperature=0)


class DeepSeekModel(BaseModel):
    """ 深度求索模型 """

    def get_reasoner_model(self):
        return ChatDeepSeek(model=DEEPSEEK_PLANNER_MODEL.get("model-name"),
                            api_key=DEEPSEEK_PLANNER_MODEL.get("api-key"),
                            api_base=DEEPSEEK_PLANNER_MODEL.get("base-url"),
                            temperature=0)

    async def stream(self, inputs):
        yield self.get_reasoner_model().stream(inputs)

    def get_model(self):
        return ChatDeepSeek(model=DEEPSEEK_WRITER_MODEL.get("model-name"),
                            api_key=DEEPSEEK_WRITER_MODEL.get("api-key"),
                            api_base=DEEPSEEK_WRITER_MODEL.get("base-url"),
                            temperature=0)
