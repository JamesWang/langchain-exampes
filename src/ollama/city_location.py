from pydantic import BaseModel

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider

class CityLocation(BaseModel):
    city: str
    country: str
    
ollma_model = OpenAIChatModel(
    model_name="qwen2.5:7b-instruct",
    provider=OllamaProvider(
        base_url="http://192.168.0.195:11434/v1",
        api_key="ollama"
    )
)

agent = Agent(ollma_model, output_type=CityLocation)
result = agent.run_sync("Where were the olympics held in 2012?")
print(result.output)
print(result.usage())