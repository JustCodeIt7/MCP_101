from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStdio
from mcp import ClientSession, StdioServerParameters
import os
# server = MCPServerStdio(
#     'uv',
#     args=[
#         'run',
#         '-N',
#         '-R=node_modules',
#         '-W=node_modules',
#         '--node-modules-dir=auto',
#         'jsr:@pydantic/mcp-run-python',
#         'stdio',
#     ]
# )

server = MCPServerStdio(
            command='uv',
            args=[
                '--directory',
                './',
                'run',
                './mcp_server.py'
            ]
        )
# server = StdioServerParameters(
#     command='uv', args=['run', 'mcp_server.py', 'server'], env=os.environ
# )
agent = Agent('openai:gpt-4.1-nano', mcp_servers=[server])


async def main():
    async with agent.run_mcp_servers():
        result = await agent.run('How many days between 2000-01-01 and 2025-03-18?')
    print(result.output)
    #> There are 9,208 days between January 1, 2000, and March 18, 2025.