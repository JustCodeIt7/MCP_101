from fastmcp import Client

async def main():
    # Connect via stdio to a local script
    async with Client("mcp_server.py") as client:
        tools = await client.list_tools()
        print(f"Available tools: {tools}")
        # result = await client.call_tool("add", {"a": 5, "b": 3})
        # print(f"Result: {result.text}")

    # Connect via SSE
    async with Client("http://localhost:8000/sse") as client:
        # ... use the client
        pass