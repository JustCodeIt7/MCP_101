# mcp_pydanticai_client.py

import asyncio
from dataclasses import dataclass

from pydantic_ai import Agent, Tool
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

@dataclass
class MCPClient:
    session: ClientSession

    async def get_hello_world(self) -> str:
        """Call the hello://world resource."""
        content, _ = await self.session.read_resource("hello://world")
        return content

    async def greet_with_prompt(self, name: str) -> str:
        """Invoke the hello_prompt prompt."""
        resp = await self.session.get_prompt(
            "hello_prompt", arguments={"name": name}
        )
        return str(resp)

    async def greet_with_tool(self, name: str) -> str:
        """Call the hello_tool tool."""
        resp = await self.session.call_tool(
            "hello_tool", arguments={"name": name}
        )
        return str(resp)

async def main():
    # 1. Configure and launch the MCP server over stdio
    server_params = StdioServerParameters(
        command="python",
        args=["server.py"],    # Make sure server.py is next to this script
    )

    async with stdio_client(server_params) as (reader, writer):
        async with ClientSession(reader, writer) as session:
            await session.initialize()

            # 2. Wrap in our helper
            mcp_client = MCPClient(session=session)

            # 3. Set up your Ollama model via PydanticAI
            ollama_model = OpenAIModel(
                model_name="llama3.2",
                provider=OpenAIProvider(base_url="http://localhost:11434/v1")
            )

            agent = Agent(
                ollama_model,
                tools=[
                    Tool(
                        name="get_hello_world",
                        func=mcp_client.get_hello_world,
                        description="Fetch the hello://world resource."
                    ),
                    Tool(
                        name="hello_prompt",
                        func=mcp_client.greet_with_prompt,
                        description="Use the hello_prompt to greet a user by name."
                    ),
                    Tool(
                        name="hello_tool",
                        func=mcp_client.greet_with_tool,
                        description="Use the hello_tool to greet a user by name."
                    ),
                ],
            )

            # 4. Demonstrate each capability
            res1 = await agent.run("Fetch the greeting from the resource.")
            print("Resource →", res1)

            res2 = await agent.run("Now greet Alice using the prompt tool.")
            print("Prompt  →", res2)

            res3 = await agent.run("Finally, greet Bob using the hello_tool.")
            print("Tool    →", res3)

if __name__ == "__main__":
    asyncio.run(main())
