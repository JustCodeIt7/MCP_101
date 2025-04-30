import asyncio
import json
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
# from langchain_community.chat_models import ChatOllama
from langchain_ollama import ChatOllama
# Mock implementation if mcp.client.aio is unavailable
class AsyncStdIOClient:
    # Ensure the module is installed or accessible
    pass
    async def connect(self, command):
        print(f"Mock: Connecting with command {command}")

    async def get_schema(self):
        print("Mock: Fetching schema")
        return {"tools": {}, "resources": {}}

    async def get_resource(self, uri):
        print(f"Mock: Accessing resource {uri}")
        return {}

# Ensure all imports are at the top of the file
from mcp.client.langchain import MCPTool

# --- Configuration ---
MCP_SERVER_COMMAND = ["python", "M04-Example_MCP_Servers/02-User_Profile_MCP_Server/server.py"]
# --- Configuration ---
MCP_SERVER_COMMAND = ["python", "M04-Example_MCP_Servers/02-User_Profile_MCP_Server/server.py"]
OLLAMA_MODEL = "llama3.2" # Make sure this model is available in your Ollama setup

# --- MCP Client Setup ---
async def get_mcp_tools():
    """Connects to the MCP server and returns LangChain tools."""
    print("Connecting to MCP server...")
    client = AsyncStdIOClient()
    await client.connect(MCP_SERVER_COMMAND)
    print("Fetching MCP schema...")
    schema = await client.get_schema()
    print("MCP Schema fetched.")

    mcp_tools = []
    if schema:
        print("Available MCP Tools:")
        for tool_name, tool_schema in schema.get("tools", {}).items():
            print(f"- {tool_name}: {tool_schema.get('description', 'No description')}")
            mcp_tools.append(MCPTool(client=client, tool_name=tool_name, tool_schema=tool_schema))
        
        # Note: Langchain agents don't directly support MCP resources yet.
        # You would need custom handling or wrap resource access in a tool if needed.
        print("\nAvailable MCP Resources:")
        for resource_uri, resource_schema in schema.get("resources", {}).items():
             print(f"- {resource_uri}: {resource_schema.get('description', 'No description')}")

    else:
        print("Warning: Could not fetch schema from MCP server.")

    # Add a simple tool to demonstrate resource access if needed
    @tool
    async def get_user_profile_resource(user_id: str) -> dict:
        """Gets a user's profile using the MCP resource."""
        uri = f"users://{user_id}/profile"
        print(f"Accessing resource: {uri}")
        try:
            profile = await client.get_resource(uri)
            return profile
        except Exception as e:
            return {"error": f"Failed to get resource {uri}: {e}"}

    @tool
    async def get_all_users_resource() -> dict:
        """Gets all user profiles using the MCP resource."""
        uri = "users://all"
        print(f"Accessing resource: {uri}")
        try:
            profiles = await client.get_resource(uri)
            return profiles
        except Exception as e:
            return {"error": f"Failed to get resource {uri}: {e}"}

    mcp_tools.extend([get_user_profile_resource, get_all_users_resource])

    return client, mcp_tools

# --- LangChain Agent Setup ---
def create_agent(llm, tools):
    """Creates a LangChain agent with the given LLM and tools."""
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant that can interact with a user profile database using the provided tools. Use the tools to answer user requests about adding, removing, or retrieving user information."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    agent = prompt | llm.bind_tools(tools)
    return agent

async def run_agent_loop(agent, tools):
    """Runs the main interaction loop for the agent."""
    chat_history = []
    print("\n--- User Profile Agent ---")
    print("Ask me about user profiles (e.g., 'Who is alice?', 'Add user dave ...', 'Remove bob', 'List all users'). Type 'quit' to exit.")

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            break

        chat_history.append(HumanMessage(content=user_input))
        agent_scratchpad = []

        # Agent invocation loop to handle tool calls
        while True:
            ai_msg = await agent.ainvoke({"input": user_input, "chat_history": chat_history, "agent_scratchpad": agent_scratchpad})

            if not ai_msg.tool_calls:
                # No tool calls, LLM responded directly
                print(f"AI: {ai_msg.content}")
                chat_history.append(ai_msg)
                break # Exit tool call loop, wait for next user input

            # Process tool calls
            agent_scratchpad.append(ai_msg) # Add AI message with tool calls
            tool_messages = []
            for tool_call in ai_msg.tool_calls:
                selected_tool = {t.name: t for t in tools}.get(tool_call["name"])
                if selected_tool:
                    print(f"AI -> Calling Tool: {tool_call['name']} with args {tool_call['args']}")
                    try:
                        tool_output = await selected_tool.ainvoke(tool_call["args"])
                        print(f"Tool -> Response: {tool_output}")
                        tool_messages.append(ToolMessage(content=json.dumps(tool_output), tool_call_id=tool_call["id"]))
                    except Exception as e:
                        print(f"Tool -> Error: {e}")
                        tool_messages.append(ToolMessage(content=json.dumps({"error": str(e)}), tool_call_id=tool_call["id"]))
                else:
                    print(f"AI -> Error: Tool '{tool_call['name']}' not found.")
                    tool_messages.append(ToolMessage(content=json.dumps({"error": f"Tool '{tool_call['name']}' not found."}), tool_call_id=tool_call["id"]))

            agent_scratchpad.extend(tool_messages) # Add tool responses

            # Let the agent process the tool results in the next iteration of this inner loop

# --- Main Execution ---
async def main():
    mcp_client = None
    try:
        mcp_client, mcp_tools = await get_mcp_tools()

        if not mcp_tools:
            print("No MCP tools found. Exiting.")
            return

        print(f"\nInitializing Ollama LLM ({OLLAMA_MODEL})...")
        llm = ChatOllama(model=OLLAMA_MODEL, temperature=0)
        print("LLM Initialized.")

        print("Creating LangChain agent...")
        agent = create_agent(llm, mcp_tools)
        print("Agent created.")

        await run_agent_loop(agent, mcp_tools)

    except Exception as e:
        print(f"\nAn error occurred: {e}")
    finally:
        if mcp_client:
            print("\nDisconnecting from MCP server...")
            await mcp_client.disconnect()
            print("Disconnected.")

if __name__ == "__main__":
    asyncio.run(main())
