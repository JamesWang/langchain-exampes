from langgraph.func import task, entrypoint
import model_init


@task
def call_llm_1(llm, topic: str):
    """First LLM call to generate initial joke"""
    msg = llm.invoke(f"Write a joke about {topic}")
    return msg.content


@task
def call_llm_2(llm, topic: str):
    """Second LLM call to generate story"""
    msg = llm.invoke(f"Write a story about {topic}")
    return msg.content


@task
def call_llm_3(llm, topic):
    """Third LLM call to generate poem"""
    msg = llm.invoke(f"Write a poem about {topic}")
    return msg.content


@task
def aggregator(topic: str, joke: str, story: str, poem: str):
    combined = f"Here is a story, joke, and poem about {topic}:\n\n"
    combined += f"Story:\n{story}\n\n"
    combined += f"Joke:\n{joke}\n\n"
    combined += f"Poem:\n{poem}\n"
    return combined


@entrypoint()
def parallel_content_workflow(inputs: dict):
    topic = inputs["topic"]
    llm = inputs["llm"]
    joke_fut = call_llm_1(llm, topic)
    story_fut = call_llm_2(llm, topic)
    poem_fut = call_llm_3(llm, topic)
    
    return aggregator(
        topic, 
        joke_fut.result(), 
        story_fut.result(), 
        poem_fut.result()
    ).result()
    

for step in parallel_content_workflow.stream(
    input={
        "llm": model_init.prepare_model(), 
        "topic": "cat"
    }, 
    stream_mode="updates"
):
    print(step)
    print("\n")