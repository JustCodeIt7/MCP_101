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

mail_server = MCPServerHTTP(url='http://localhost:8000/sse')

# Ollama Agent
ollama_model = OpenAIModel(
    model_name='llama3.2', provider=OpenAIProvider(base_url='http://eos-parkmour.netbird.cloud:11434/v1')
)

agent = Agent(ollama_model, mcp_servers=[brave_search_mcp_server])



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