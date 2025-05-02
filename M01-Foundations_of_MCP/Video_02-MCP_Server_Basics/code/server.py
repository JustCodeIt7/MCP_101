# hello_world_server.py

# Video Script Point:
# 1. Explain the MCP Python SDK and FastMCP class
# 2. Show how to create a server instance
# 3. Add a basic Resource and a basic Tool
# 4. Run the server
import logging

from mcp.server.fastmcp import FastMCP
# logging.getLogger('mcp').setLevel(logging.WARNING)

# Create an MCP server named "HelloWorld"
# FastMCP handles connections, protocol compliance, and message routing :contentReference[oaicite:0]{index=0}&#8203;:contentReference[oaicite:1]{index=1}
mcp = FastMCP("HelloWorld")

# --- 1. Define a Resource that returns "Hello, world!" ---
# Resources behave like read-only endpoints (no side effects) :contentReference[oaicite:2]{index=2}&#8203;:contentReference[oaicite:3]{index=3}
@mcp.resource("hello://world")
def hello_resource() -> str:
    """Return a simple greeting."""
    return "Hello, world!"

# --- 2. Define a Tool that also returns "Hello, world!" ---
# Tools can perform computation or have side effects :contentReference[oaicite:4]{index=4}&#8203;:contentReference[oaicite:5]{index=5}
@mcp.tool()
def hello_tool() -> str:
    """Tool that returns the same greeting."""
    return "Hello, world!"

# --- 3. Run the server when executed directly ---
if __name__ == "__main__":
    # Starts the MCP server over SSE on the default port (8000)
    # You can then connect with the CLI: `mcp dev hello_world_server.py` :contentReference[oaicite:6]{index=6}&#8203;:contentReference[oaicite:7]{index=7}
    mcp.run()
