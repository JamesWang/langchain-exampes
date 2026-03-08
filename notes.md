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
