import chainlit as cl
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()


@cl.on_message
async def chat(message: cl.Message):
    session_id = cl.user_session.get("id")
    thread = {"configurable": {"thread_id": session_id}}
    messages = cl.chat_context.get()
    # TODO 执行智能体调度



if __name__ == '__main__':
    from chainlit.cli import run_chainlit

    run_chainlit(__file__)
