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

if __name__ == "__main__":
    mcp.run(transport="stdio")