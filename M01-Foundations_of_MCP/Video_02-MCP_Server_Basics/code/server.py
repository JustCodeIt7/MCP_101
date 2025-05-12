import logging

from mcp.server.fastmcp import FastMCP

logging.getLogger("mcp").setLevel(logging.WARNING)

# Create an MCP server named "HelloWorld"
# FastMCP handles connections, protocol compliance, and message routing
mcp = FastMCP("HelloWorld")

# --- 1. Define a Resource that returns "Hello, world!" ---
# Resources expose data to LLMs like GET endpoints in a REST API, providing information without complex computations or side effects.
@mcp.resource("hello://world")
def hello_resource() -> str:
    """Return a simple greeting."""
    # e.g. call an external API, or do some complex math
    # return the result
    return "Hello, world!"

# --- 2. Prompt the user to enter a name ---
# Prompts are reusable templates that help LLMs interact with your server effectively:
@mcp.prompt()
def hello_prompt(name: str) -> str:
    """Prompt to greet a user by name."""
    # Prompt template to greet the user
    # This is a simple example, but you can use more complex templates
    prompt_template = f"Hello, {name}!"
    return prompt_template

# --- 3. Define a Tool that also returns "Hello, world!" ---
# Tools enable LLMs to execute actions via your server, performing computations and generating side effects beyond passive resource retrieval.
@mcp.tool()
def hello_tool(name:str) -> str:
    """Tool that returns the same greeting."""
    # do some computation here
    # name = name[:2]
    # create a new file, or send an email
    return f"Hello, world! {name}"
