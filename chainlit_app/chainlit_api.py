from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, AIMessageChunk, ToolMessage
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
    async for chunk in deep_researcher.astream(
            input={
                "messages": messages,
            },
            context={
                "thread_id": message.thread_id,
                "session_id": cl.user_session.get("id"),
            },
            stream_mode=["updates"],
            subgraphs=True,
    ):
        logger.info(chunk)
        graph_node, stream_mode, node = chunk
        if not graph_node:
            # 说明在主图节点
            for node_name, node_value in node.items():
                if node_name == "clarify_with_user":
                        response = node_value.get("messages")[-1]
                        await cl.Message(content=response.content).send()
                if node_name == "write_research_brief":
                    research_brief = node_value.get("research_brief")
                    await cl.Message(content="研究简介：\n\n" + research_brief).send()
        else:
            # 具体的子图节点
            sub_graph_node = graph_node[-1]
            if sub_graph_node.startswith("research_supervisor:"):
                for node_name, node_value in node.items():
                    if node_name == "supervisor":
                        response = node_value.get("supervisor_messages")[-1]
                        if isinstance(response, AIMessage):
                            if response.content:
                                await cl.Message(content=response.content).send()
                            if response.tool_calls:
                                tool_call = response.tool_calls[0]
                                async with cl.Step(name=tool_call.get("name"),
                                                   type="tool",
                                                   id=tool_call.get("id"),
                                                   default_open=True) as research_topic_tool:
                                    research_topic_tool.input = tool_call.get("args")
                    elif node_name == "researcher":
                        pass
            elif sub_graph_node.startswith("supervisor_tools:"):
                if node_name == "supervisor_tools":
                    response = node_value.get("supervisor_messages")[-1]
                    if isinstance(response, ToolMessage):
                        async with cl.Step(name=response.name, type="tool", id=response.tool_call_id,
                                           default_open=True) as step:
                            step.output = response.content





