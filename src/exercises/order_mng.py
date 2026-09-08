from langchain_openai.chat_models import ChatOpenAI
from langchain_core.messages import SystemMessage, ToolMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END, START

from typing import TypedDict, Optional
import logging
import sys


from tools import tools_list, tool_lookup_map, tool_node

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] (SERVER) [%(filename)s:%(lineno)d] %(message)s",
    handlers=[
        logging.FileHandler("app.log", mode="a"),
        logging.StreamHandler(sys.stderr)  # <-- CRITICAL: Keeps stdout 100% pure JSON
    ],
    force=True
)
logger = logging.getLogger(__name__)

MODEL_NAME="qwen2.5-7b-instruct" #"qwen/qwen3-4b-thinking-2507"
MODEL_API_KEY="lm-studio"
AI_BASE_URL="http://localhost:1234/v1"


class MyAgentState(TypedDict):
    order: Optional[str]
    messages: list


def call_model(state):
    msgs = state["messages"]
    order = state.get("order", {"order_id": "UNKNOWN"})

    prompt = (
        f"You are an e-commerce customer support agent.\n"
        f"CURRENT CONTEXT - ORDER ID: {order['order_id']}\n\n"
        f"INSTRUCTIONS:\n"
        f"1. Use the appropriate tool matching the customer's request if available.\n"
        f"2. For all other requests, respond normally and politely."
    )

    full = [SystemMessage(prompt)] + msgs
    model = ChatOpenAI(
        base_url=AI_BASE_URL,
        model=MODEL_NAME,
        api_key=MODEL_API_KEY, 
        temperature=0
    )

    # 1. Bind the dynamic tools list natively
    model_with_tools = model.bind_tools(tools_list)
    first = model_with_tools.invoke(full)
    node_outputs = [first]

    if getattr(first, "tool_calls", None):
        logger.info("----------- Processing Dynamic Tool Call -------------")
        tc = first.tool_calls[0]
        requested_tool_name = tc["name"]
        
        if requested_tool_name in tool_lookup_map:
            active_tool = tool_lookup_map[requested_tool_name]
            result = active_tool.invoke(tc["args"])
            
            # Append historical record blocks
            node_outputs.append(ToolMessage(content=result, tool_call_id=tc["id"]))
            
            # Append final message block directly
            final_response = AIMessage(content=result)
            node_outputs.append(final_response)
        else:
            error_msg = f"Error: Tool '{requested_tool_name}' is not registered."
            node_outputs.append(AIMessage(content=error_msg))
            
    return {"messages": node_outputs}


def should_continue(state: MyAgentState):
    messages_field = state["messages"]
    
    # If your node returned a nested list, extract the first inner item block safely
    if isinstance(messages_field, list) and len(messages_field) > 0 and isinstance(messages_field[-1], list):
        last_message = messages_field[-1][-1] # Pull the message out of the nested array
    else:
        last_message = messages_field[-1]
        
    if getattr(last_message, "tool_calls", None):
        return "tools"
    return END


def construct_graph():
    workflow = StateGraph(MyAgentState)
    workflow.add_node("assistant", call_model)
    workflow.add_node("tools", tool_node)
    workflow.add_edge(START, "assistant")    
    workflow.add_conditional_edges("assistant", should_continue)

    workflow.add_edge("tools", "assistant")
    return workflow.compile()


graph = construct_graph()

if __name__ == "__main__":
    example_order = {"order_id": "A12345"}
    convo = [HumanMessage(content="Please cancel my order A12345.")]
    result = graph.invoke({"order": example_order, "messages": convo})
    logger.info(f'result={result}')
    for msg in result["messages"]:
        print(f"{msg.type}: {msg.content}")