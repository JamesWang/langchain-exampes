import sqlite3
from typing import TypedDict

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, interrupt

class FormState(TypedDict):
    age: int | None
    

def get_age_node(state: FormState) -> FormState:
    prompt  = "What is your age? "
    while True:
        answer =  interrupt(prompt)
        if isinstance(answer, int) and answer > 0:
            return {"age": answer}
        prompt = f"'{answer}' is not a valid age. Please enter a positive integer for your age: "


builder = (
    StateGraph(FormState)
    .add_node("get_age", get_age_node)
    .add_edge(START, "get_age")
    .add_edge("get_age", END)
)
# Note: the check_same_thread needs to be set to False `here` to avoid 
#    sqlite3.ProgrammingError: SQLite objects created in a thread can only be used in that same thread.
checkpointer = SqliteSaver(sqlite3.connect("form_state.db", check_same_thread=False))
graph = builder.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "form-1"}}
first = graph.invoke({"age":None}, config=config)
print(first['__interrupt__'])

retry = graph.invoke(Command(resume="thirty"), config=config)
print(retry['__interrupt__'])

final = graph.invoke(Command(resume=30), config=config)
print(final['age'])

