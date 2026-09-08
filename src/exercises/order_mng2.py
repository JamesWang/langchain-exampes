import logging
import sys
from typing import Annotated
from typing_extensions import TypedDict


from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

from tools import tools_list, tool_lookup_map, tool_node

MODEL_NAME="qwen/qwen-2.5-7b-instruct" #"qwen/qwen3-4b-thinking-2507"
MODEL_API_KEY="lm-studio"
AI_BASE_URL="http://localhost:1234/v1"

# 1. Setup explicit logging straight to standard error
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stderr,
    force=True
)
logger = logging.getLogger(__name__)


model = ChatOpenAI(
    base_url=AI_BASE_URL, 
    model=MODEL_NAME, # A standard instruct model works best
    api_key=MODEL_API_KEY,
    temperature=0
)
model_with_tools = model.bind_tools(tools_list)

# 5. Define LangGraph State Structure
class AgentState(TypedDict):
    messages: Annotated[list, lambda x, y: x + y]

# 6. Define the Workflow Nodes
def call_model(state: AgentState):
    """The Agent node that decides whether to speak or call a tool."""
    response = model_with_tools.invoke(state["messages"])
    return {"messages": [response]}

def should_continue(state: AgentState):
    """Router logic that checks if the LLM requested a tool call."""
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools" # Routes to the ToolNode execution block
    return END # Ends the workflow and returns to user

# 7. Construct and Compile the StateGraph
workflow = StateGraph(AgentState)

# Add our active execution nodes
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

# Map the layout paths
workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue)
workflow.add_edge("tools", "agent") # Routes back to agent to summarize tool outputs

app = workflow.compile()

# 8. Test Execution Run
if __name__ == "__main__":
    inputs = {"messages": [("user", "Please cancel my order A9876 right now.")]}
    
    print("\n--- Starting Graph Execution ---\n")
    final_state = app.invoke(inputs)
    print("\n--- Final Conversational Answer ---")
    print(final_state["messages"][-1].content)
