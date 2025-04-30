import asyncio
import traceback
from typing import Any, Dict, List

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.prebuilt import create_react_agent
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools

# Choose and configure your language model
# from langchain_openai import ChatOpenAI
# model = ChatOpenAI(model="gpt-4o")
from langchain_ollama import ChatOllama
from rich import print


def create_server_params(script_path: str) -> StdioServerParameters:
    """Creates StdioServerParameters for a given MCP server script."""
    # Ensure the script_path is absolute for reliability
    import os

    absolute_script_path = os.path.abspath(script_path)
    print(f"Using server script: {absolute_script_path}")
    return StdioServerParameters(
        command="python",
        args=[absolute_script_path],
    )


async def connect_and_load_tools(server_params: StdioServerParameters) -> tuple[ClientSession, List[BaseTool]]:
    """
    Connects to the MCP server, initializes the session, and loads tools.

    Returns a tuple containing the active ClientSession and the loaded tools.
    Handles the context management for stdio_client and ClientSession internally,
    so the caller needs to manage the returned session's lifecycle if needed outside
    the initial tool loading. For simplicity in this refactor, we assume the session
    is primarily needed for tool loading within this scope.
    A more robust implementation might return the context managers or require
    the caller to handle them.
    """
    # This approach simplifies the main logic but means the session
    # is tied to the lifetime of this specific operation.
    # If the session is needed for longer, the context managers
    # should be handled in the calling function (e.g., test_mcp_server).
    # For this refactoring, we'll load tools and assume the session isn't
    # needed further by the direct caller of this function.

    # Re-thinking: It's better to establish the connection and session
    # in the main testing function to manage the context properly.
    # This function will just load tools given an active session.

    # Let's adjust the plan: connect_and_load_tools is not the right abstraction.
    # The connection needs to wrap the agent invocation.

    # New plan: Keep connection logic in the main test function.
    # Create helpers for agent creation/invocation and response printing.
    pass  # This function is removed in favor of handling context in the main test function.


async def run_agent_query(model: BaseChatModel, tools: List[BaseTool], query: str) -> Dict[str, Any]:
    """Creates a ReAct agent, invokes it with the query, and returns the response."""
    print("Creating ReAct agent...")
    agent = create_react_agent(model, tools)
    print(f"Invoking agent with query: '{query}'")
    agent_response = await agent.ainvoke({"messages": [("user", query)]})  # Pass query as user message
    print("Agent invocation complete.")
    return agent_response


def print_agent_response(agent_response: Dict[str, Any]):
    """Prints the messages from the agent response in a structured format."""
    print("\n--- Agent Response Messages ---")
    if not agent_response or "messages" not in agent_response:
        print("No messages found in the response.")
        return

    for message in agent_response["messages"]:
        role = "Unknown"
        content = ""
        tool_info = ""
        tool_calls_info = ""

        # Extract Role
        if hasattr(message, "__class__"):
            role = message.__class__.__name__.replace("Message", "")

        # Extract Content
        if hasattr(message, "content") and message.content:
            content = message.content

        # Extract Tool Name (for ToolMessage)
        if hasattr(message, "name") and message.name:
            # Check if it's a ToolMessage by checking for tool_call_id
            if hasattr(message, "tool_call_id"):
                tool_info = f"Tool Result for: {message.name}"
            # Otherwise, it might be part of an AIMessage's tool_calls (handled below)

        # Extract Tool Calls (for AIMessage)
        if hasattr(message, "tool_calls") and message.tool_calls:
            calls = []
            for tool_call in message.tool_calls:
                call_name = tool_call.get("name", "unknown_tool")
                call_args = tool_call.get("args", {})
                calls.append(f"Call: {call_name}(args={call_args})")
            if calls:
                tool_calls_info = f"Tool Calls: {'; '.join(calls)}"

        # Print formatted message
        print(f"\n[{role}]:")
        if content:
            print(f"  Content: {content}")
        if tool_info:
            print(f"  {tool_info}")
        if tool_calls_info:
            print(f"  {tool_calls_info}")


async def test_mcp_server(server_script_path: str, model: BaseChatModel, query: str):
    """
    Connects to an MCP server, runs a query using a ReAct agent, and prints the response.

    Args:
        server_script_path: The path to the MCP server Python script.
        model: The language model instance (e.g., ChatOllama, ChatOpenAI).
        query: The natural language query for the agent.
    """
    print("\n--- Starting MCP Server Test ---")
    print(f"Server Script: {server_script_path}")
    print(f"Model: {model.__class__.__name__}")
    print(f"Query: '{query}'")

    server_params = create_server_params(server_script_path)

    try:
        async with stdio_client(server_params) as (read, write):
            print("MCP stdio client connected.")
            async with ClientSession(read, write) as session:
                print("MCP ClientSession established.")
                # Initialize the connection
                await session.initialize()
                print("MCP Session Initialized.")

                # Get tools
                tools = await load_mcp_tools(session)
                tool_names = [tool.name for tool in tools]
                print(f"Loaded Tools: {tool_names}")
                if not tools:
                    print("Warning: No tools loaded from the MCP server.")
                    # Decide if you want to proceed without tools or exit
                    # return

                # Create agent, run query, and get response
                agent_response = await run_agent_query(model, tools, query)

                # Print the response details
                print_agent_response(agent_response)

    except asyncio.TimeoutError:
        print("\n--- Error: Connection or operation timed out. ---")
        print("Ensure the server script is running correctly and accessible.")
    except ConnectionRefusedError:
        print("\n--- Error: Connection refused. ---")
        print("Ensure the server process can be started and is not blocked.")
    except Exception as e:
        print("\n--- An unexpected error occurred ---")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Details: {e}")
        print("Traceback:")
        traceback.print_exc()
    finally:
        print("\n--- MCP Server Test Finished ---")


# --- Configuration ---
# Choose the model
# model = ChatOpenAI(model="gpt-4o") # Requires OPENAI_API_KEY
model = ChatOllama(model="llama3.2")  # Requires Ollama server running

# Define the server and query to test
# Option 1: User Profile Server
SERVER_SCRIPT = "/Users/james/Github/YouTube/MCP_101/M04-Example_MCP_Servers/02-User_Profile_MCP_Server/server.py"
QUERY_TO_RUN = "What is the user's name and what is their favorite color?"

# Option 2: Basic Calculator Server
# SERVER_SCRIPT = "/Users/james/Github/YouTube/MCP_101/M04-Example_MCP_Servers/01-Basic_Calculator_MCP_Server/server.py"
# QUERY_TO_RUN = "what's (3 + 5) * 12?"

# --- Main Execution ---
if __name__ == "__main__":
    # Ensure the SERVER_SCRIPT path is correct before running
    import os

    if not os.path.exists(SERVER_SCRIPT):
        print(f"Error: Server script not found at {SERVER_SCRIPT}")
        print("Please update the SERVER_SCRIPT variable with the correct absolute path.")
    else:
        asyncio.run(test_mcp_server(SERVER_SCRIPT, model, QUERY_TO_RUN))