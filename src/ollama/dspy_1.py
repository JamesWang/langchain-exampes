import dspy

"""_summary_
This example demonstrates how to use the dspy library to interact with a language model (LLM) hosted on an Ollama server. The code initializes the LLM with the specified model and API details, configures dspy to use this LLM, and then sends a simple query to the model to get a response.
Note:
 - model parameter specifies the provider and model name in the format "provider/model_name:version".
    * ollama_chat indicates that the model is hosted on an Ollama server and is a chat-based model.
    * qwen2.5:7b-instruct specifies the model name and version.
- api_base is the URL of the Ollama server where the model is hosted.
- api_key is left empty in this example, but it can be set if the Ollama server requires authentication.
"""
llm = dspy.LM(
    model="ollama_chat/qwen2.5:7b-instruct",
    api_base='http://192.168.0.195:11434', 
    api_key=''
)

dspy.configure(lm=llm)

print(llm("What is 2 + 2?"))
