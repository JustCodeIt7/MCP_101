from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStdio, MCPServerHTTP
from dotenv import load_dotenv
import asyncio
import os

from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

load_dotenv()

brave_search_mcp_server = MCPServerStdio(
    command= "npx",
    args=[
          "-y",
          "@modelcontextprotocol/server-brave-search"
        ],
    env= {"BRAVE_API_KEY": os.getenv("BRAVE_API_KEY")}
)


# mail_server = MCPServerHTTP(url='http://localhost:8000/sse')

#stdio mode
mail_server = MCPServerStdio(
    command= "uv",
    args= [
        "run",
        "--with",
        "mcp[cli]",
        "mcp",
        "run",
        "/Users/james/Github/YouTube/MCP_101/M01-Foundations_of_MCP/Video_03-Setting_Up_Your_First_MCP_Server/code/mcp_server/mcp_server.py"
      ]
)
# Ollama Agent
ollama_model = OpenAIModel(
    model_name='qwen3:4b', provider=OpenAIProvider(base_url='http://eos-parkmour.netbird.cloud:11434/v1')
)

agent = Agent(ollama_model, mcp_servers=[brave_search_mcp_server, mail_server])



async def main():
    async with agent.run_mcp_servers():
        while True:
            command = input("You: ")
            if command == "exit":
                break
            agent_response = await agent.run(command)
            print(f"Agent: {agent_response}")


if __name__ == "__main__":
    asyncio.run(main())