from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOllama(
    model="qwen2.5:7b-instruct", # Or your preferred model
    base_url="http://192.168.0.195:11434", # Replace with your remote IP
    temperature=0
)



# 1. Define a simple tool
@tool
def get_word_length(word: str) -> int:
    """Returns the length of a word."""
    return len(word)

tools = [get_word_length]

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant that uses tools when necessary."),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)

# 2. Construct the agent
agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="You are a helpful assistant that uses tools when necessary."
)

# 4. Run the agent
result = agent.invoke({"messages": "What is the length of the word 'LangChain'?"})

final_message = result["messages"][-1]
print("-----------------------------------------------")
print(f"{result}")
print("-----------------------------------------------")
print(f"len={final_message.content}")
