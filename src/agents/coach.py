# agents/coach.py
"""
Progress Coach — supervisor that orchestrates Planner, Explainer, and the
A2A Quiz Generator, with SQLite checkpointing and human-in-the-loop.
"""
import asyncio
from functools import wraps
import inspect
import json
import re
import uuid
from typing import Callable, Optional, TypedDict
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import interrupt, Command
import httpx

from agents.planner import build_planner_graph
from agents.explainer import build_explainer_graph

A2A_QUIZ_URL = "http://localhost:9001/"


class CoachState(TypedDict):
    topic: str
    plan: list[dict]
    current_index: int
    explanation: str
    quiz: dict
    answers: list[dict]
    
def print_it(func):
    def log_call(action, name):
        print(f"{'-> Start' if action == 'start' else '<- End'}: {name}")
    
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        log_call("start", func.__name__)
        try:
            v = await func(*args, **kwargs)
            print("-" * 100)
            print(v)
            print("-" * 100)    
            return v
        finally:
            log_call("end", func.__name__)

    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        log_call("start", func.__name__)
        try:
            v = func(*args, **kwargs)
            print("-" * 100)
            print(v)
            print("-" * 100)    
            return v
        finally:
            log_call("end", func.__name__)
    
    return async_wrapper if inspect.iscoroutinefunction(func) else sync_wrapper


async def call_quiz(payload: dict) -> str:
    """Tiny inlined A2A client. In production use the official a2a SDK client."""
    body = {
        "jsonrpc": "2.0",
        "id": str(uuid.uuid4()),
        "method": "SendMessage",  # 🟢 Changed from message/send
        "params": {
            "message": {                
                "role": "ROLE_USER",  # 🟢 Changed from user
                "parts": [
                    {
                        # 🟢 v1.0 utilizes 'root' union wrappers for parts
                        "root": {
                            "text": json.dumps(payload)
                        }
                    }
                ],
                "messageId": str(uuid.uuid4()),
            }
        },        
    }
    headers = {
        "Content-Type": "application/json",
        "X-A2A-Version": "1.0",
        "A2A-Version": "1.0"
    }
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(A2A_QUIZ_URL, json=body, headers=headers)
        r.raise_for_status()
        data = r.json()
    # Drill into the JSON-RPC response to find the agent's text reply.
    #print(data)
    try:
        message_payload = data["result"]["message"]
        parts = message_payload["parts"]
    
        first_part = parts[0]
        if "root" in first_part and "text" in first_part["root"]:
            print(first_part["root"]["text"])
            return first_part["root"]["text"]
        elif "text" in first_part:  # Fallback in case a lightweight gateway flattened it
            print(first_part["text"])
            return first_part["text"]
            
        raise KeyError("Could not find a valid text component inside the parts list.")

    except KeyError as e:
        print(f"Extraction failed. Full payload received was: {data}")
        raise e

@print_it
async def plan_topic(state: CoachState) -> CoachState:
    g = build_planner_graph()
    out = await asyncio.to_thread(g.invoke, {"topic": state["topic"], "plan": []})
    return {"plan": out["plan"], "current_index": 0} # type: ignore


@print_it
async def explain_item(state: CoachState) -> CoachState:    
    plan_list = state.get("plan", [])
    idx = state.get("current_index", 0)
    if not plan_list or idx >= len(plan_list):
        return {} # type: ignore

    topic_node = plan_list[idx]
    title = topic_node.get("title", "Unknown Topic")
    
    print(f"-> Explaining Topic: {title}")
    
    g = await build_explainer_graph()
    out = await g.ainvoke({"item_title": title, "messages": [], "explanation": ""})
    
    return {"explanation": out["explanation"]} # type: ignore


@print_it
async def make_quiz(state: CoachState) -> CoachState:    
    raw_text = await call_quiz({"action": "generate", "explanation": state["explanation"]})
    raw_blocks = re.findall(r"\{\s*\"q\":.+?\}\s*(?=\s*(?:,|\}\s*$|\]))", raw_text)

    clean_blocks = []
    for block in raw_blocks:
        # Strip trailing brackets or junk Qwen hallucinations from the end of each block
        block = block.strip()
        # Count the braces to ensure it closes properly with exactly one '}'
        if block.endswith("}}"):
            block = block[:-1]
        elif not block.endswith("}"):
            block += "}"

        clean_blocks.append(block)

    # Cleanly rebuild the JSON string from scratch
    clean_json_str = '{"questions": [' + ", ".join(clean_blocks) + "]}"

    return {"quiz": json.loads(clean_json_str)} # type: ignore


@print_it
async def ask_learner(state: CoachState) -> dict:  # Changed type hint to dict
    """Human-in-the-loop: pause and wait for the learner's answer."""
    # 1. Safely pull the active quiz question
    quiz_obj = state.get("quiz")
    if not quiz_obj or not quiz_obj.get("questions"):
        # If no question exists, exit safely to prevent crashes
        return {}

    first_q = quiz_obj["questions"][0]
    
    # 2. This creates the interrupt hook. 
    # When you call Command(resume=choice), execution resumes EXACTLY on this line.
    answer = interrupt({
        "kind": "ask_learner",
        "question": first_q["q"],
        "options": first_q["options"],
    })
    
    # 3. Grade the choice that was passed into Command(resume)
    feedback = await call_quiz({
        "action": "grade",
        "question": first_q["q"],
        "answer": answer,
        "correct": first_q["answer"],
    })
    
    # 4. Compile the updated feedback list
    current_answers = state.get("answers") or []
    updated_answers = current_answers + [{
        "question": first_q["q"],
        "given": answer,
        "correct": first_q["answer"],
        "feedback": feedback,
    }]

    # 5. CRITICAL: Explicitly clear the "quiz" field out of the state dictionary.
    # This prevents the graph from instantly re-processing the same question 
    # when it cycles back around.
    return {
        "answers": updated_answers,
        "quiz": None  
    }


@print_it
def decide_next(state: CoachState) -> str:
    current_idx = state.get("current_index", 0)
    plan_list = state.get("plan", [])

    if current_idx >= len(plan_list) - 1:
        print("--- TOPIC PLAN COMPLETE: ROUTING TO END ---")
        return "done"

    print(f"--- MOVING TO NEXT ITEM: {current_idx + 1} OF {len(plan_list)} ---")
    return "next_item"


@print_it
async def advance(state: CoachState) -> CoachState:    
    return {
        "current_index": state["current_index"] + 1, 
        "explanation": "", 
        "quiz": {}
    } # pyright: ignore[reportReturnType]


async def build_coach():
    g = StateGraph(CoachState)
    g.add_node("plan", plan_topic)
    g.add_node("explain", explain_item)
    g.add_node("quiz", make_quiz)
    g.add_node("ask", ask_learner)
    g.add_node("advance", advance)
    g.set_entry_point("plan")
    g.add_edge("plan", "explain")
    g.add_edge("explain", "quiz")
    g.add_edge("quiz", "ask")
    g.add_conditional_edges("ask", decide_next, {"next_item": "advance", "done": END})
    g.add_edge("advance", "explain")

    saver = AsyncSqliteSaver.from_conn_string("./coach.sqlite")
    return g, saver


async def main():
    g, saver_cm = await build_coach()
    async with saver_cm as saver:
        graph = g.compile(checkpointer=saver)
        config: RunnableConfig | None = {"configurable": {"thread_id": "session-001"}}

        # First run — will pause at the first interrupt() asking for an answer.
        result = await graph.ainvoke(
            {"topic": "LangGraph for beginners", "plan": [], "current_index": 0,
             "explanation": "", "quiz": {}, "answers": []},
            config=config,
        )

        while True:
            state_info = await graph.aget_state(config)
            if not state_info.tasks:
                break

            if state_info.tasks[0].interrupts:                
                ask = state_info.tasks[0].interrupts[0].value
                print(f"\nQ: {ask['question']}")
                for i, opt in enumerate(ask["options"]):
                    print(f"  {chr(65 + i)}. {opt}")
                    
                choice = input("Your answer (A/B/C/D): ").strip().upper()
                result = await graph.ainvoke(Command(resume=choice), config=config)
            else:
                break
        print("\n--- Session summary ---")
        for a in result["answers"]:
            print(f"Q: {a['question']}\n  you: {a['given']} | correct: {a['correct']}\n  feedback: {a['feedback']}\n")


if __name__ == "__main__":
    asyncio.run(main())