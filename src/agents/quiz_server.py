# agents/quiz_server.py
"""
Quiz Generator exposed as an A2A server on http://localhost:9001.
The agent card lives at http://localhost:9001/.well-known/agent-card.json
"""
import json
from typing import cast
import uuid
from langchain_core.messages import AIMessage
import uvicorn
from starlette.applications import Starlette

# ✅ Updated v1.x router imports (A2AStarletteApplication was removed)
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCard, AgentCapabilities, AgentInterface, AgentSkill, Message, Part, Role
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.helpers import new_text_message

#from a2a.utils import new_agent_text_message
from llm_factory import get_llm

GEN_PROMPT = """Write 3 multiple-choice questions to test understanding of:

{explanation}

Return your entire response strictly as a single JSON object matching the exact structure below. 
Do not close the array early, and ensure every question object is nested inside the single "questions" array.

Example structure:
{{
  "questions": [
    {{
      "q": "What is the result of 2 + 2?",
      "options": ["A) 3", "B) 4", "C) 5", "D) 6"],
      "answer": "B"
    }},
    {{
      "q": "Which keyword is used to define a function in Python?",
      "options": ["A) func", "B) define", "C) def", "D) function"],
      "answer": "C"
    }}
  ]
}}

Do not wrap it in markdown codeblocks (like ```json). Return raw text only."""


GRADE_PROMPT = """The learner answered '{answer}' to the question '{question}'.
The correct answer is '{correct}'. Reply with one short, encouraging sentence
that confirms or gently corrects them."""



class QuizExecutor(AgentExecutor):
    async def execute(self, context: RequestContext, event_queue: EventQueue):
        text = context.get_user_input() or ""
        try:
            req = json.loads(text)
        except Exception:
            req = {"action": "generate", "explanation": text}

        action = req.get("action", "generate")

        if action == "generate":
            # ✅ Disabled json_mode=True to prevent LM Studio 400 parameter errors
            llm = get_llm(temperature=0.4)
            resp: AIMessage = llm.invoke(GEN_PROMPT.format(explanation=req["explanation"]))
            await event_queue.enqueue_event(new_text_message(str(resp.content)))
            return

        if action == "grade":
            llm = get_llm(temperature=0.1)
            resp = llm.invoke(GRADE_PROMPT.format(
                answer=req["answer"],
                question=req["question"],
                correct=req["correct"],
            ))
            await event_queue.enqueue_event(new_text_message(str(resp.content))) # type: ignore
            return

        await event_queue.enqueue_event(
            new_text_message(json.dumps({"error": "unknown action"}))
        )

    async def cancel(self, context, event_queue):
        pass


def build_app() -> Starlette:
    skills = [
        AgentSkill(
            id="quiz_generate",
            name="Generate a quiz",
            description="Create 3 multiple-choice questions from an explanation.",
            tags=["quiz", "generate"],
            examples=['{"action":"generate","explanation":"..."}'],
        ),
        AgentSkill(
            id="quiz_grade",
            name="Grade an answer",
            description="Mark a learner's answer with encouraging feedback.",
            tags=["quiz", "grade"],
            examples=['{"action":"grade","question":"...","answer":"B","correct":"A"}'],
        ),
    ]
    """
    A2A is deliberately small. It does not specify how the agent thinks, which framework it uses, 
    or how it stores state — those are private implementation details. It only specifies the wire 
    contract two agents need to share to cooperate. That minimalism is what makes A2A useful across 
    frameworks; LangGraph, CrewAI, AutoGen and dozens of other libraries can all expose A2A endpoints 
    because A2A asks nothing about their internals.
    """
    card = AgentCard(
        name="Quiz Generator",
        description="Generates and grades short quizzes.",        
        version="1.0.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=False),
        skills=skills,
        supported_interfaces=[AgentInterface(
            url="http://localhost:90001",
            protocol_binding="http-json-rpc", # or your version's specific enum/string
            protocol_version="1.0"
        )
    ]

    )
    handler = DefaultRequestHandler(
        agent_card=card,
        agent_executor=QuizExecutor(),
        task_store=InMemoryTaskStore(),
    )
    
    # ✅ Explicitly build the Starlette App using the modern A2A route constructors
    routes = []
    routes.extend(create_agent_card_routes(card))
    routes.extend(create_jsonrpc_routes(handler, rpc_url="/"))
    
    return Starlette(routes=routes)


if __name__ == "__main__":
    uvicorn.run(build_app(), host="127.0.0.1", port=9001)
