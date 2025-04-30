from mcp.server.fastmcp import FastMCP

# Instantiate FastMCP
mcp = FastMCP("demo_server")

@mcp.tool()
def echo(msg: str) -> str:
    """Echo back the input message."""
    return msg

@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b

@mcp.tool()
def subtract(a: int, b: int) -> int:
    """Subtract b from a."""
    return a - b

@mcp.tool()
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b

@mcp.tool()
def divide(a: int, b: int) -> float:
    """Divide a by b."""
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b

if __name__ == "__main__":
    mcp.run(transport="stdio")