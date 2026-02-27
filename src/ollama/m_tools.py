from functools import partial
from typing import Literal
from langchain.messages import ToolMessage
from model_init import MessageState, llm_invoke, prepare_model
from tools import add, divide, multiply

model = prepare_model()

tools = [multiply, add, divide]
tools_by_name = {tool.name: tool for tool in tools}
model_with_tools = model.bind_tools(tools)

def llm_call(state: dict):
    return llm_invoke(llm=model_with_tools, state=state)

    
def tool_node(state: dict):
    """Performs the tool call
    AIMessage(
        content="",
        tool_calls=[{
            'name': 'get_weather', 
            'args': {'location': 'Toronto'}, 
            'id': 'call_12345'
        }]
    )
    
    state["messages"] = [
        HumanMessage(content="What is the weather in Toronto?"),
        AIMessage(
            content="",  # Often empty when calling tools
            tool_calls=[
                {
                    "name": "get_weather", 
                    "args": {"location": "Toronto"}, 
                    "id": "call_99xYv"  # Unique ID for this specific call
                }
            ]
        )
    ]

    """
    result = []
    for tool_call in state["messages"][-1].tool_calls: # all the last message's tool calls
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
    return {"messages": result}

from langgraph.graph import StateGraph, START, END

def should_continue(state: MessageState) -> Literal["tool_node", END]:
    """
    Decide if we should continue the loop or stop based upon the presence of tool calls in the last message.
    """
    messages = state["messages"]
    last_message = messages[-1] if messages else None
    
    if last_message.tool_calls:
        return "tool_node"
    return END

agent_builder = StateGraph(MessageState)
# build graph
# START -> llm_call -> (tool_node or END) -> (llm_call or END)
agent = agent_builder \
    .add_node("llm_call", llm_call) \
    .add_node("tool_node", tool_node) \
    .add_edge(START,"llm_call") \
    .add_conditional_edges(
        "llm_call", 
        should_continue,
        ["tool_node", END]
    ) \
    .add_edge("tool_node", "llm_call") \
    .compile()

#from IPython.display import display, Image
#display(Image(agent.get_graph(xray=TRue).draw_mermaid_png()))

from langchain.messages import HumanMessage
messages = [HumanMessage(content="Add 3 and 4")]
messages = agent.invoke({"messages": messages})
for m in messages["messages"]:
    m.pretty_print()
