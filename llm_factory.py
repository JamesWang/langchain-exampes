# llm_factory.py
"""
A tiny helper that returns a LangChain chat model based on LLM_PROVIDER.
Beginners: you do not need to read every line — you just need to know
the rest of the project calls get_llm(temperature=...) and gets back
a ready-to-use chat model.
"""
import os
from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel

load_dotenv()

PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower()
model_name = "qwen2.5-7b-instruct"
AI_BASE_URL="http://localhost:1234/v1"

def get_llm(temperature: float = 0.2, json_mode: bool = False) -> BaseChatModel:
    if PROVIDER == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        kwargs = {"model": "gemini-2.0-flash", "temperature": temperature}
        if json_mode:
            kwargs["model_kwargs"] = {"response_mime_type": "application/json"}
        return ChatGoogleGenerativeAI(**kwargs)

    if PROVIDER == "groq":
        from langchain_groq import ChatGroq
        kwargs = {"model": "llama-3.3-70b-versatile", "temperature": temperature}
        if json_mode:
            kwargs["model_kwargs"] = {"response_format": {"type": "json_object"}}
        return ChatGroq(**kwargs)

    if PROVIDER == "openai":
        from langchain_openai import ChatOpenAI
        kwargs = {
            "model": model_name,
            "base_url": AI_BASE_URL,
            "temperature": temperature
        }
        if json_mode:
            # Nest it directly inside model_kwargs to stop the LangChain parameters warning
            kwargs["model_kwargs"] = {
                "response_format": {"type": "text"}
            }

        return ChatOpenAI(**kwargs)

    raise ValueError(f"Unknown LLM_PROVIDER: {PROVIDER}")
