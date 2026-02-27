
import operator
from typing import Annotated, TypedDict
from langchain.chat_models import init_chat_model
from langchain.messages import AnyMessage, SystemMessage


class MessageState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int
    

def prepare_model():
    return init_chat_model(
        model="qwen2.5:7b-instruct", # Or your preferred model
        model_provider="ollama",
        base_url="http://192.168.0.195:11434",
        temperature=0
    )
    

def llm_invoke(llm, state: dict):
    """LLM decides whether to call a tool or respond directly based on the conversation history.

    Args:
        state (dict): _description_
    """
    _system_prompt = "You are a helpful assistant that can perform basic math operations using tools when necessary."
    return {
        "messages": [
            llm.invoke(
                input=[SystemMessage(content=_system_prompt)] + state["messages"]
            )
        ],
        "llm_calls": state.get("llm_calls", 0) + 1
    }