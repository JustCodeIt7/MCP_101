import asyncio
import traceback
from typing import Any, Dict, List
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.prebuilt import create_react_agent
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from rich import print
import os

# Choose and configure your language model
# from langchain_openai import ChatOpenAI
# model = ChatOpenAI(model="gpt-4o")
from langchain_ollama import ChatOllama


def create_server_params(script_path: str) -> StdioServerParameters:
    """Creates StdioServerParameters for a given MCP server script."""
    # Ensure the script_path is absolute for reliability
    absolute_script_path = os.path.abspath(script_path)
    print(f"Using server script: {absolute_script_path}")
    return StdioServerParameters(
        command="python",
        args=[absolute_script_path],
    )


async def run_agent_query(model: BaseChatModel, tools: List[BaseTool], query: str) -> Dict[str, Any]:
    """Creates a ReAct agent, invokes it with the query, and returns the response."""
    print("\n" + "="*50)
    print("CREATING AGENT".center(50))
    print("="*50)
    print("• Creating ReAct agent with model:", model.__class__.__name__)
    print("• Available tools:", len(tools))
    print("\n" + "-"*50)
    print("EXECUTING QUERY".center(50))
    print("-"*50)
    print(f"Query: \"{query}\"")
    
    agent = create_react_agent(model, tools)
    agent_response = await agent.ainvoke({"messages": [("user", query)]})
    
    print("\n" + "-"*50)
    print("EXECUTION COMPLETE".center(50))
    print("-"*50)
    return agent_response


def print_agent_response(agent_response: Dict[str, Any]):
    """Prints the messages from the agent response in a structured format."""
    print("\n" + "="*80)
    print("AGENT RESPONSE MESSAGES".center(80))
    print("="*80)
    
    if not agent_response or "messages" not in agent_response:
        print("❌ No messages found in the response.")
        return

    for i, message in enumerate(agent_response["messages"]):
        print("\n" + "-"*80)
        
        role = "Unknown"
        if hasattr(message, "__class__"):
            role = message.__class__.__name__.replace("Message", "")
        print(f"MESSAGE #{i+1} | TYPE: {role}")
        print("-"*80)
        
        # Extract Content
        if hasattr(message, "content") and message.content:
            print("CONTENT:")
            print(f"{message.content}")
            print()

        # Extract Tool Name (for ToolMessage)
        if hasattr(message, "name") and message.name:
            if hasattr(message, "tool_call_id"):
                print("TOOL RESULT:")
                print(f"Tool: {message.name}")
                print()

        # Extract Tool Calls (for AIMessage)
        if hasattr(message, "tool_calls") and message.tool_calls:
            print("TOOL CALLS:")
            for j, tool_call in enumerate(message.tool_calls):
                call_name = tool_call.get("name", "unknown_tool")
                call_args = tool_call.get("args", {})
                print(f"  Call #{j+1}: {call_name}")
                print(f"  Arguments: {call_args}")
                print()


async def test_mcp_server(server_script_path: str, model: BaseChatModel, queries: List[str]):
    """
    Connects to an MCP server, runs multiple queries using a ReAct agent, and prints the responses.

    Args:
        server_script_path: The path to the MCP server Python script.
        model: The language model instance (e.g., ChatOllama, ChatOpenAI).
        queries: A list of natural language queries for the agent.
    """
    print("\n" + "="*80)
    print("MCP SERVER TEST STARTING".center(80))
    print("="*80)
    print("• Server Script: " + server_script_path)
    print("• Model: " + model.__class__.__name__)
    print("• Number of Queries: " + str(len(queries)))

    server_params = create_server_params(server_script_path)

    try:
        print("\n" + "-"*50)
        print("CONNECTING TO MCP SERVER".center(50))
        print("-"*50)
        
        async with stdio_client(server_params) as (read, write):
            print("✓ MCP stdio client connected successfully")
            async with ClientSession(read, write) as session:
                print("✓ MCP ClientSession established")
                # Initialize the connection
                await session.initialize()
                print("✓ MCP Session Initialized")

                # Get tools
                print("\n" + "-"*50)
                print("LOADING TOOLS".center(50))
                print("-"*50)
                tools = await load_mcp_tools(session)
                if not tools:
                    print("⚠ WARNING: No tools loaded from the MCP server")
                else:
                    print(f"✓ Successfully loaded {len(tools)} tools:")
                    for i, tool in enumerate(tools):
                        print(f"  {i+1}. {tool.name}")

                # Process each query in the list
                for i, query in enumerate(queries):
                    print("\n" + "="*80)
                    print(f"PROCESSING QUERY #{i+1}".center(80))
                    print("="*80)
                    print(f"Query: \"{query}\"")
                    
                    # Create agent, run query, and get response
                    agent_response = await run_agent_query(model, tools, query)

                    # Print the response details
                    print_agent_response(agent_response)

    except asyncio.TimeoutError:
        print("\n" + "="*80)
        print("ERROR: CONNECTION TIMEOUT".center(80))
        print("="*80)
        print("The connection or operation timed out.")
        print("Please ensure the server script is running correctly and accessible.")
    except ConnectionRefusedError:
        print("\n" + "="*80)
        print("ERROR: CONNECTION REFUSED".center(80))
        print("="*80)
        print("The connection was refused by the server.")
        print("Please ensure the server process can be started and is not blocked.")
    except Exception as e:
        print("\n" + "="*80)
        print("ERROR: UNEXPECTED EXCEPTION".center(80))
        print("="*80)
        print(f"Type: {type(e).__name__}")
        print(f"Details: {e}")
        print("\nTraceback:")
        traceback.print_exc()
    finally:
        print("\n" + "="*80)
        print("MCP SERVER TEST COMPLETE".center(80))
        print("="*80)


# --- Main Execution ---
if __name__ == "__main__":
    # --- Configuration ---
    # model = ChatOpenAI(model="gpt-4o") # Requires OPENAI_API_KEY
    # model = ChatOllama(model="qwen3:4b")     
    # model = ChatOllama(model="llama3.2")     
    model = ChatOllama(model="phi4-mini", temperature=0.3)     
    
    # Option 1: Basic Calculator Server
    SERVER_SCRIPT = "/Users/james/Github/YouTube/MCP_101/M04-Example_MCP_Servers/01-Basic_Calculator_MCP_Server/server.py"
    QUERIES_TO_RUN = [
        "what's (3 + 5) * 12?",
        "Calculate the square root of 144",
        "If I have 50 and spend 15, then multiply by 2, what do I have?"
    ]
    
    # Option 2: User Profile Server
    # SERVER_SCRIPT = "/Users/james/Github/YouTube/MCP_101/M04-Example_MCP_Servers/02-User_Profile_MCP_Server/server.py"
    # QUERIES_TO_RUN = [
    #     "What is the user's name and what is their favorite color?",
    #     "How old is the user and what are their hobbies?",
    #     "Tell me about the user's job and location"
    # ]
    
    # Ensure the SERVER_SCRIPT path is correct before running
    if not os.path.exists(SERVER_SCRIPT):
        print("\n" + "="*80)
        print("ERROR: SERVER SCRIPT NOT FOUND".center(80))
        print("="*80)
        print(f"The server script was not found at: {SERVER_SCRIPT}")
        print("Please update the SERVER_SCRIPT variable with the correct absolute path.")
    else:
        asyncio.run(test_mcp_server(SERVER_SCRIPT, model, QUERIES_TO_RUN))
