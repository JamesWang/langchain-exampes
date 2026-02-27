from langchain.tools import tool

@tool
def multiply(a: int, b: int) -> int:
    """Multiplies two numbers."""
    return a * b

@tool
def add(a: int, b: int) -> int:
    """Adds two numbers."""
    return a + b


@tool
def divide(a: int, b: int) -> int:
    """Divides two numbers."""
    if b == 0:
        return "Error: Division by zero is undefined."
    return a / b
