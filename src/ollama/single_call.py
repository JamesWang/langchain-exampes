from typing import Annotated, TypedDict
from langchain_ollama import ChatOllama
from langchain.tools import tool
from langchain_core.messages import BaseMessage
from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode


# -------------------------
# 1️⃣ Define Tool
# -------------------------
@tool
def calculator(expression: str) -> str:
    """Evaluate a math expression."""
    return str(eval(expression))


# -------------------------
# 2️⃣ Define State
# -------------------------
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# -------------------------
# 3️⃣ LLM (Ollama)
# -------------------------
llm = ChatOllama(
    model="qwen2.5:7b-instruct", # Or your preferred model
    base_url="http://192.168.0.195:11434", # Replace with your remote IP
    temperature=0
)

llm_with_tools = llm.bind_tools([calculator])


# -------------------------
# 4️⃣ Chatbot Node
# -------------------------
def chatbot(state: AgentState):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


# -------------------------
# 5️⃣ Tool Node (Terminal)
# -------------------------
def run_tool_and_stop(state: AgentState):
    last_message = state["messages"][-1]

    if not last_message.tool_calls:
        return {"messages": []}

    tool_call = last_message.tool_calls[0]
    tool_name = tool_call["name"]
    tool_args = tool_call["args"]

    if tool_name == "calculator":
        result = calculator.invoke(tool_args)

        # Return final AI-style message directly
        return {
            "messages": [
                AIMessage(content=result)
            ]
        }

    return {"messages": []}


# -------------------------
# 6️⃣ Build Graph
# -------------------------
graph = StateGraph(AgentState)

graph.add_node("chatbot", chatbot)
graph.add_node("tools", run_tool_and_stop)

graph.set_entry_point("chatbot")

# If tool call → go to tools and END
graph.add_conditional_edges(
    "chatbot",
    lambda state: "tools" if state["messages"][-1].tool_calls else END,
)

# IMPORTANT: tools go directly to END
graph.add_edge("tools", END)

app = graph.compile()


# -------------------------
# 7️⃣ Run
# -------------------------
result = app.invoke(
    {"messages": [("user", "What is 25 * 4?")]}
)

print(result["messages"][-1].content)