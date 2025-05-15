import asyncio
import os

import logging
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from rich import print
from pydantic_ai import Agent

logging.getLogger("mcp").setLevel(logging.WARNING)


# create a new agent




async def client():
    server_params = StdioServerParameters(
        command='uv', args=['run', 'mcp_server.py', 'server'], env=os.environ
    )
    
    # agent = Agent('openai:gpt-4.1-nano', mcp_servers=[server_params])
    agent = Agent('openai:gpt-4.1-nano')

    # Connect to the server
    # async with stdio_client(server_params) as (read, write):
    #     async with ClientSession(read, write) as session:
    #         # Send a ping request
    #         await session.initialize()
    #
    #         # list all tools
    #         print('\n########################### Tool List ###########################')
    #         tools = await session.list_tools()
    #         print('\n', tools)
    #
    #         # use the tool
    #         print('\n########################### Tool Response ###########################')
    #         tool_response = await session.call_tool('poet', {'theme': 'socks'})
    #         print('\n', tool_response)
    #
    #         # list all resources
    #         print('\n########################### Resource List ###########################')
    #         resources = await session.list_resources()
    #         print('\n', resources)
    #         # use the resource
    #         print('\n########################### Resource Content ###########################')
    #         content, mime_type = await session.read_resource('hello://world')
    #         print('\n', content)
    #         print('\n########################### Resource Mime Type ###########################')
    #         print('\n', mime_type)
    #
    #         # list resource templates
    #         print('\n########################### Resource Templates List ###########################')
    #         resource_templates = await session.list_resource_templates()
    #         print('\n', resource_templates)
    #
    #         # list all prompts
    #         print('\n########################### Prompt List ###########################')
    #         prompts = await session.list_prompts()
    #         print('\n', prompts)
    #
    #         # use the prompt with the agent to get a response
    #         print('\n########################### Using MCP in Pydantic AI Agent ###########################')
    #


    async with agent.run_mcp_servers():
        result = await agent.run('How many days between 2000-01-01 and 2025-03-18?')
        print(result.output)
            
            
if __name__ == '__main__':
    asyncio.run(client())
