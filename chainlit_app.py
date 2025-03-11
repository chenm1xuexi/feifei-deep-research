import chainlit as cl
from langgraph.checkpoint.memory import MemorySaver

from deep_research.graph import report_builder

memory = MemorySaver()
workflow = report_builder.compile(checkpointer=memory)


@cl.on_chat_start
async def on_chat_start():
    await cl.Message(content="你好，我是小飞飞，请输入你想要研究的主题").send()


@cl.on_message
async def chat(message: cl.Message):
    session_id = cl.user_session.get("id")
    thread = {"configurable": {"thread_id": session_id}}

    user_chat_history = [message for message in cl.chat_context.get() if message.type == 'user_message']
    if len(user_chat_history) == 1:
        topic = message.content
        async for event in workflow.astream({"topic": topic}, thread, stream_mode="updates"):
            # 所有输出展示结果统一 交给 chainlit 去渲染展示，因此这里只是启动graph，无需其他调度
            pass


if __name__ == '__main__':
    from chainlit.cli import run_chainlit

    run_chainlit(__file__)
