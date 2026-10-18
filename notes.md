### Langraph interrupt

- **Rules of interrupts**
  - [ ] When you call interrupt within a node, LangGraph suspends execution by raising an exception that signals the runtime to pause. This exception propagates up through the call stack and is caught by the runtime, which notifies the graph to save the current state and wait for external input.

  - [ ] When execution resumes (after you provide the requested input), the runtime **restarts the entire node from the beginning**—it **does not resume from the exact line where interrupt was called**. This means any code that ran before the interrupt will execute again. Because of this, there’s a few important rules to follow when working with interrupts to ensure they behave as expected.
​
  - [ ] **Do not wrap interrupt calls in try/except**
    The way that interrupt pauses execution at the point of the call is **by throwing a special exception**. If you wrap the interrupt call in a try/except block, you will catch this exception and the interrupt will not be passed back to the graph.

> [!IMPORTANT]
> ✅ **Separate interrupt calls from error-prone code**
> ✅ **Use specific exception types in try/except blocks**

###

### 🛠️ The Rule of Thumb for LangGraph Design

| Scenario | Best Router Choice | Reason |
| :--- | :--- | :--- |
| **Clear, structured rules** (e.g., API status codes, counter limits, environment flags, boolean toggles) | **Predefined Python Router** | Fast, cheap, and bulletproof. |
| **Unstructured, fuzzy rules** (e.g., Categorizing a user's raw text email into "angry", "billing question", or "technical bug") | **LLM-Based Router** | Required because code cannot easily parse human intent without natural language understanding. |

---

### LangGraph Conditional Routing Cheat Sheet

In LangGraph, graph.add_conditional_edges() is used to create dynamic routing in your state machine. It allows the graph to look at the current state (or environment) and decide which node to visit next, rather than following a fixed, linear path.

📋 Syntax Breakdown

```python
graph.add_conditional_edges(
    source,
    path,
    path_map
)
```

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`source`** | `str` | The name of the node *from which* the conditional routing starts. |
| **`path`** | `callable` | **The Router Function.** A Python function that takes the current `State` as input and returns a string indicating the routing decision. |
| **`path_map`** | `dict` | A mapping dictionary matching the output string of your `path` function to the actual next node name defined in your graph. |

---

How the Decision is Made (Step-by-Step)

1. Execution: The source node finishes its execution and updates the graph's State.
2. Evaluation: LangGraph automatically intercepts the workflow and calls your path router function, passing the updated State into it.
3. The Output: Your router function executes its logic (Python rules or an LLM call) and returns a custom string representing the decision.
4. The Jump: LangGraph looks up that string inside the path_map dictionary to find the corresponding target node name, and routes the workflow there.

🛠️ Implementation Patterns

Pattern 1: Programmatic / Deterministic Router (High Efficiency & Zero Cost)

Best for clear, structured rules (e.g., status codes, counter limits, environment flags, boolean toggles).
python
import os
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

### Define State

class GraphState(TypedDict):
    retry_count: int
    api_status: int

### Predefined Python Router (Deterministic)

def route_by_system_rules(state: GraphState) -> str:
    if state["api_status"] == 200:
        return "success"
    elif state["retry_count"] >= 3:
        return "exhausted"
    else:
        return "retry"

### Graph Configuration

```python
workflow = StateGraph(GraphState)

workflow.add_conditional_edges(
    "api_caller_node",       # Source node
    route_by_system_rules,   # Router function
    {
        "success": "processor_node",
        "retry": "api_caller_node",  # Loop back
        "exhausted": "alert_human_node"
    }
)
```

Pattern 2: LLM Intent Router (Handling Unpredictable User Input)

Best when you cannot predict exactly how a human will phrase a request, requiring natural language understanding to sort into categories.
python
from pydantic import BaseModel, Field
from typing import Literal

### Define Structured Output for the LLM

```python
class RouteChoice(BaseModel):
    intent: Literal["billing", "technical", "general"] = Field(
        description="Categorize user intent based on their message."
    )
```

### Predefined LLM Router

```python
  def user_intent_router(state: GraphState) -> str:
      # Use structured output to force the LLM to choose a valid key
      structured_llm = model.with_structured_output(RouteChoice)
      last_message = state["messages"][-1].content

      decision = structured_llm.invoke(last_message)
      return decision.intent

workflow.add_conditional_edges(
    "input_receiver_node",
    user_intent_router,
    {
        "billing": "billing_team_node",
        "technical": "tech_support_node",
        "general": "chitchat_node"
    }
)
```

Pattern 3: LLM Evaluation Router (Self-Correction & Quality Loops)

Best for agentic review setups where a secondary critic LLM must evaluate the output of a prior step (e.g., grading code syntax or essay quality) before moving forward.
python
class QualityGrading(BaseModel):
    is_satisfactory: bool = Field(description="True if output meets criteria, False otherwise.")

### Predefined Self-Correction Router

```python
def quality_control_router(state: GraphState) -> str:
    structured_llm = model.with_structured_output(QualityGrading)
    grading = structured_llm.invoke(f"Critique this draft: {state['draft']}")

    if grading.is_satisfactory:
        return "pass"
    return "fail"

# Graph Configuration

workflow.add_conditional_edges(
    "writer_node",
    quality_control_router,
    {
        "pass": "publisher_node",
        "fail": "writer_node"  # Loops back to rewrite based on state feedback
    }
)
```

## ⚖️ The Developer's Matrix

When building LangGraph applications, keeping this mental framework prevents over-engineering:

| Is the data source predictable? | Router Type | Cost/Latency |
| :--- | :--- | :--- |
| **Yes** (Status codes, variables, counters) | **Python Native** (`if/else`) | Near-zero |
| **No** (User utterances, AI-generated essays) | **LLM-Based** (Structured output) | Moderate |

---

## ⚖️ Router Selection Matrix

| Router Class | Best Used For | Execution Latency | Cost | Reliability |
| :--- | :--- | :--- | :--- | :--- |
| **Deterministic Code** | System metrics, counters, environment flags, API codes. | Microseconds | $0.00 | 100% Predictable |
| **LLM Structured Output** | Unpredictable user input, unstructured text analysis, AI output grading. | Milliseconds to Seconds | Variable (Token-based) | High (but depends on prompt engineering) |

---