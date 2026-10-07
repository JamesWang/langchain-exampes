# agents/planner.py
"""
Curriculum Planner — the simplest possible LangGraph.
One node, JSON output, no tools.
"""
from typing import List, Sequence, TypedDict, cast
import json
from langchain_core.language_models import LanguageModelInput
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langgraph.graph import StateGraph, END
from openai import BaseModel
from pydantic import Field
from llm_factory import get_llm


class PlanSchema(BaseModel):
    topic: str = Field(description="The original topic provided.")
    plan: List[dict] # = Field(description="An ordered list of steps to execute.")


class PlannerState(TypedDict):
    topic: str
    plan: list[dict]


PROMPT = """You are a friendly teacher building a study plan for a curious beginner.
The learner wants to study: {topic}

Return a JSON object with a single key "items" whose value is a list of 3 to 5
short study items. Each item must have:
  - "title": a 4-8 word title
  - "summary": one sentence describing what the learner will gain
  - "difficulty": one of "starter", "intermediate", "advanced"

Return ONLY valid JSON. No prose, no markdown fences."""

base_llm = get_llm(temperature=0.2, json_mode=True)
structured_llm = cast(
    Runnable[LanguageModelInput, PlanSchema],
    base_llm.with_structured_output(PlanSchema)
)

def plan_node(state: PlannerState) -> PlannerState:
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert research and project planning agent."),
        ("human", PROMPT)
    ])
    formatted_prompt = prompt.format_messages(topic=state["topic"])    
    
    response: PlanSchema = structured_llm.invoke(formatted_prompt)

    return {"topic": state["topic"], "plan": response.plan}


def build_planner_graph():
    g = StateGraph(PlannerState)
    g.add_node("plan", plan_node)
    g.set_entry_point("plan")
    g.add_edge("plan", END)
    return g.compile()


if __name__ == "__main__":
    graph = build_planner_graph()
    out = graph.invoke({"topic": "LangGraph for beginners", "plan": []})
    print(json.dumps(out["plan"], indent=2))