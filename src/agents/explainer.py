# agents/explainer.py
"""
Explainer — LangGraph agent that reads local notes via MCP, then writes
a beginner-friendly explanation.
"""
import asyncio
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from llm_factory import get_llm


class ExplainerState(TypedDict):
    item_title: str
    messages: Annotated[list, add_messages] # doing things like Write Monad, collect all the messages with appending
    explanation: str


SYSTEM = """You are a patient teacher writing a 200-word beginner explanation
of '{item}'. First, call the available filesystem tools to look for any
existing notes the learner has on this topic; if you find any, weave their
ideas into your explanation. Always end with a single short example."""


async def build_explainer_graph():
    client = MultiServerMCPClient({
        "filesystem": {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-filesystem", "./notes"],
            "transport": "stdio",
        }
    })
    """
    MultiServerMCPClient introspects them, generates LangChain Tool objects with the correct argument schemas, 
    and hands them to bind_tools so the LLM can choose them.
    Without MCP you would write a Python read_file function for the explainer, then duplicate it (or refactor it) 
    when the planner needs filesystem access too, then again when a third agent in a different language joins. 
    With MCP, the filesystem server is one process, written once, that any agent in any language can use; 
    adding a fourth agent is zero extra tool code. This is precisely the value proposition that pushed MCP to 
    industry-standard status during 2025
    """
    tools = await client.get_tools()
    llm = get_llm(temperature=0.3).bind_tools(tools)
    
    async def model_node(state: ExplainerState) -> dict:
        # 1. Prepare the sequence of messages for the LLM call
        if not state["messages"]:
            sys = SystemMessage(content=SYSTEM.format(item=state["item_title"]))
            human = HumanMessage(content=f"Please explain: {state['item_title']}")
            input_msgs = [sys, human]
            
            # This is the delta we must return if it's the very first turn
            # because sys and human don't exist in the state yet!
            new_messages_delta = [sys, human] 
        else:
            input_msgs = state["messages"]
            new_messages_delta = []

        # 2. Invoke the model
        ai_msg = await llm.ainvoke(input_msgs)
        
        # 3. Add the brand new AI message to our turn payload delta
        new_messages_delta.append(ai_msg)

        # 4. Monadic state composition payload
        update = {"messages": new_messages_delta}
        
        if not ai_msg.tool_calls:
            update["explanation"] = ai_msg.content # type: ignore
            
        return update

    g = StateGraph(ExplainerState)
    g.add_node("model", model_node)
    g.add_node("tools", ToolNode(tools))
    g.set_entry_point("model")
    g.add_conditional_edges("model", tools_condition)
    g.add_edge("tools", "model")
    return g.compile()

async def main():
    graph = await build_explainer_graph()
    out = await graph.ainvoke({
        "item_title": "What LangGraph is",
        "messages": [],
        "explanation": "",
    })
    print("\n--- Final explanation ---\n")
    print(out["explanation"])


if __name__ == "__main__":
    asyncio.run(main())