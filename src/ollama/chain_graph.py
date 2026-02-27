from typing import Annotated, TypedDict
from langchain_ollama import ChatOllama
from langchain.tools import tool
from langchain_core.messages import BaseMessage
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
# 3️⃣ Create LLM (Ollama)
# -------------------------
llm = ChatOllama(
    #model="qwen3:4b",
    #"gemma3:4b",
    model="qwen2.5:7b-instruct",
    base_url="http://192.168.0.195:11434", 
    #base_url="http://192.168.0.211:11434",
    temperature=0,
)

llm_with_tools = llm.bind_tools([calculator])


# -------------------------
# 4️⃣ Define Nodes
# -------------------------
def chatbot(state: AgentState):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


tool_node = ToolNode([calculator])


# -------------------------
# 5️⃣ Build Graph
# -------------------------
graph = StateGraph(AgentState)

graph.add_node("chatbot", chatbot)
graph.add_node("tools", tool_node)

graph.set_entry_point("chatbot")

graph.add_conditional_edges(
    "chatbot",
    lambda state: "tools" if state["messages"][-1].tool_calls else END,
)

graph.add_edge("tools", "chatbot")

app = graph.compile()

print("=================================\n")
print("running the agent...\n")
# -------------------------
# 6️⃣ Run
# -------------------------
result = app.invoke(
    {"messages": [("user", "What is 25 * 4?")]}
)

for m in result["messages"]:
    print(m)