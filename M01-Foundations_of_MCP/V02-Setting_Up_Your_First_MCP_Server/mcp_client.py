# mcp_client.py
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from pydantic_ai.mcp import MCPServerStdio
from rich import print
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai import Agent
from dotenv import load_dotenv
import os
load_dotenv()


basic_mcp_server = MCPServerStdio(
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
    model_name='llama3.2', provider=OpenAIProvider(base_url=os.getenv('OLLAMA_BASE_URL'))
)

agent = Agent(ollama_model, mcp_servers=[basic_mcp_server])


async def summarize_url(url: str) -> str:
    """
    Extracts text from a URL and generates a summary using the agent.

    Args:
        url: The URL to extract text from and summarize

    Returns:
        A summary of the content from the URL
    """
    async with agent.run_mcp_servers():
        # First, extract the text from the URL using the MCP tool
        extraction_prompt = (
            f"Use the extract_text_from_url tool to get content from {url}"
        )
        extraction_result = await agent.run(extraction_prompt)

        # Now, ask the agent to summarize the extracted content
        summary_prompt = (
            f"Summarize the following content in 3-5 key points:\n\n{extraction_result}"
        )
        summary = await agent.run(summary_prompt)

        return summary

async def main():
    server_params = StdioServerParameters(
        command="python",
        args=["mcp_server.py"],
    )

    async with stdio_client(server_params) as (reader, writer):
        async with ClientSession(reader, writer) as session:
            await session.initialize()

            result = await session.call_tool("add", arguments={"a": 3, "b": 4})
            print(f"Result of add tool: {result}")

            # result = await session.call_tool("extract_text_from_url", arguments={"url": "https://modelcontextprotocol.io/introduction"})
            # # print the result markdown
            # print(result.content[0].text)

            # Use the agent-based approach to summarize a URL
            url = "https://modelcontextprotocol.io/introduction"
            summary = await summarize_url(url)
            print(f"\n--- URL Summary ---\n{summary}")


if __name__ == "__main__":
    asyncio.run(main())