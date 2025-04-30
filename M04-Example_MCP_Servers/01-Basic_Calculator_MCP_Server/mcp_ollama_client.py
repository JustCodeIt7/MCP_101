# Create server parameters for stdio connection
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import asyncio
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent

from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from rich import print
# model = ChatOpenAI(model="gpt-4o")
model = ChatOllama(model="llama3.2")
server_params = StdioServerParameters(
    command="python",
    # Make sure to update to the full absolute path to your math_server.py file
    args=["/Users/james/Github/YouTube/MCP_101/M04-Example_MCP_Servers/02-User_Profile_MCP_Server/server.py"],
)

async def main():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the connection
            await session.initialize()

            # Get tools
            tools = await load_mcp_tools(session)

            # Create and run the agent
            agent = create_react_agent(model, tools)
            agent_response = await agent.ainvoke({"messages": "what's (3 + 5) x 12?"})
            
            # Extract and print just the contents from the response messages
            print("\n--- Response Messages ---")
            for message in agent_response['messages']:
                # Check if the message has content
                if hasattr(message, 'content') and message.content:
                    # Print the role (if available) and content
                    role = message.__class__.__name__.replace('Message', '')
                    print(f"\n[{role}]: {message.content}")
                
                # If it's a tool message, also print the tool name
                if hasattr(message, 'name') and message.name:
                    print(f"Tool used: {message.name}")
                    
                # If there are tool calls in the message, print those too
                if hasattr(message, 'tool_calls') and message.tool_calls:
                    for tool_call in message.tool_calls:
                        print(f"Tool call: {tool_call['name']} with args: {tool_call['args']}")

# Run the async function
asyncio.run(main())