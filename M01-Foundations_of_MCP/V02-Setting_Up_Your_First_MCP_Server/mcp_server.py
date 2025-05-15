# mcp_server.py
from mcp.server.fastmcp import FastMCP
from crawl4ai import AsyncWebCrawler # Import AsyncWebCrawler
import logging
logging.basicConfig(level=logging.WARNING)
# Create an MCP server instance with a custom name.
mcp = FastMCP("Demo Server")

# Add a calculator tool: a simple function to add two numbers.
@mcp.tool()
def add(a: int, b: int) -> int:
    """
    Add two numbers together.

    :param a: First number.
    :param b: Second number.
    :return: Sum of the numbers.
    """
    return a + b

# New tool to extract text from a URL using Crawl4AI
@mcp.tool()
async def extract_text_from_url(url: str = 'https://modelcontextprotocol.io/introduction') -> str:
    """
    Extracts text content from a given URL as markdown using Crawl4AI.

    :param url: The URL of the website to extract text from.
    :return: The extracted text content in markdown format, or an error message if extraction fails.
    """
    print(f"Extracting text from URL: {url}")
    try:
        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(url=url)
            if result and result.markdown:
                return result.markdown
            elif result and not result.markdown:
                return "Could not extract markdown content from the URL. The page might be empty or not processable."
            else:
                return "Failed to get a result from the crawler."
    except Exception as e:
        # Log the exception for debugging on the server side if needed
        # print(f"Error during crawling URL {url}: {e}")
        return f"An error occurred while trying to extract text from the URL: {str(e)}"

# Expose a greeting resource that dynamically constructs a personalized greeting.
@mcp.resource("greeting://{name}")
def get_greeting(name: str) -> str:
    """
    Return a greeting for the given name.

    :param name: The name to greet.
    :return: A personalized greeting.
    """
    return f"Hello, {name}!"

@mcp.prompt()
def review_code(code: str) -> str:
    """
    Provide a template for reviewing code.

    :param code: The code to review.
    :return: A prompt that asks the LLM to review the code.
    """
    return f"Please review this code:\n\n{code}"

if __name__ == "__main__":
    # Note: mcp.run() will handle the asyncio event loop for async tools.
    mcp.run()