from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver

from common.log import logger

import chainlit as cl

from deep_research.graph.builder import deep_researcher

memory = MemorySaver()


@cl.on_chat_start
async def main():
    await cl.Message(content="我是您专业的深度研究助手，请问有什么可以帮您？").send()


def to_langchain_message():
        messages = []
        for message in cl.chat_context.get():
            if message.type == "assistant_message":
                messages.append(AIMessage(content=message.content))
            elif message.type == "user_message":
                messages.append(HumanMessage(content=message.content))
            elif message.type == "system_message":
                messages.append(SystemMessage(content=message.content))

        return messages



@cl.on_message
async def chat(message: cl.Message):
    messages = to_langchain_message()
    print(messages)
    async for stream_mode, chunk in deep_researcher.astream(
            input={
                "messages": messages,
            },
            context={
                "thread_id": message.thread_id,
                "session_id": cl.user_session.get("id"),
            },
            stream_mode=["updates"],
    ):
        logger.info(chunk)
        if chunk.get("clarify_with_user"):
            ai_message = chunk.get("clarify_with_user").get("messages")[-1]
            await cl.Message(content=ai_message.content).send()

