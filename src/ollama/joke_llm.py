from langgraph.func import task, entrypoint
import model_init

@task
def generate_joke(llm, topic: str):
    msg = llm.invoke(f"Write a short joke about {topic}")
    return msg.content

def check_punchline(joke: str):
    # A very basic check for a punchline (for demonstration purposes)
    return "Fail" if "?" in joke or "!" in joke else "Pass"

@task
def improve_joke(llm, joke: str):
    msg = llm.invoke(f"Make this joke funnier by adding wordplay: {joke}")
    return msg.content


@task
def polish_joke(llm, joke: str):
    msg = llm.invoke(f"Add a suprise twist to this joke: {joke}")
    return msg.content

@entrypoint()
def prompt_chaining_workflow(inputs: dict):
    llm = inputs["llm"]
    topic = inputs["topic"]
    joke = generate_joke(llm, topic).result()
    print(f"Generated Joke: {joke}")
    
    if check_punchline(joke) == "Fail":
        joke = improve_joke(llm, joke).result()
        print(f"Improved Joke: {joke}")
    
    polished_joke = polish_joke(llm, joke).result()
    print(f"Polished Joke: {polished_joke}")
    
llm = model_init.prepare_model()
input={
    "llm":llm, 
    "topic":"cats"
}
for step in prompt_chaining_workflow.stream(input=input, stream_mode="updates"):
    print(step)
    print("\n")