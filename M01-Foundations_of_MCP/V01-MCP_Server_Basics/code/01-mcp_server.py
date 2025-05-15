"""
Demo MCP Server - A simple server that exposes Python functions as MCP Tools

This script demonstrates how to create a minimal MCP server using the FastMCP
class and expose Python functions as MCP Tools using the @mcp.tool() decorator.
"""

# Note: Removed 'asyncio' import as it's no longer directly used here.
# The MCP/AnyIO library will handle the async loop internally.
from typing import List, Dict, Any, Optional
from mcp.server.fastmcp import FastMCP

# Initialize the FastMCP server with a name
mcp_server = FastMCP("DemoServer")

@mcp_server.tool()
def add(a: int, b: int) -> int:
    """
    Add two numbers together and return the result.

    Args:
        a: The first number to add
        b: The second number to add

    Returns:
        The sum of the two numbers
    """
    return a + b

@mcp_server.tool()
def multiply(a: float, b: float) -> float:
    """
    Multiply two numbers together.

    Args:
        a: The first number to multiply
        b: The second number to multiply

    Returns:
        The product of the two numbers
    """
    return a * b

@mcp_server.tool()
def search_books(query: str, max_results: Optional[int] = 5) -> List[Dict[str, Any]]:
    """
    Search for books matching the given query.

    Args:
        query: The search query (e.g., "python programming", "science fiction")
        max_results: Maximum number of results to return (default: 5)

    Returns:
        A list of matching books, each represented as a dictionary with title, author, and year.
    """
    # In a real application, this might query a database or API
    # For this demo; we'll return some fake data based on the query
    sample_books = [
        {"title": "Python Crash Course", "author": "Eric Matthes", "year": 2019},
        {"title": "Fluent Python", "author": "Luciano Ramalho", "year": 2021},
        {"title": "Python for Data Analysis", "author": "Wes McKinney", "year": 2022},
        {"title": "Automate the Boring Stuff with Python", "author": "Al Sweigart", "year": 2020},
        {"title": "Python Cookbook", "author": "David Beazley", "year": 2013},
        {"title": "Effective Python", "author": "Brett Slatkin", "year": 2019},
        {"title": "Learning Python", "author": "Mark Lutz", "year": 2013},
    ]

    # Simple filtering based on query (case-insensitive)
    results = [
        book for book in sample_books
        if query.lower() in book["title"].lower() or query.lower() in book["author"].lower()
    ]

    # Limit results
    return results[:max_results]

@mcp_server.tool()
def format_text(text: str, options: Dict[str, Any] = None) -> str:
    """
    Format the given text according to the specified options.

    Args:
        text: The text to format
        options: A dictionary of formatting options which may include:
            - uppercase (bool): Convert the text to uppercase
            - lowercase (bool): Convert the text to lowercase
            - reverse (bool): Reverse the text
            - repeat (int): Number of times to repeat the text

    Returns:
        The formatted text
    """
    if options is None:
        options = {}

    result = text

    if options.get("uppercase", False):
        result = result.upper()

    if options.get("lowercase", False):
        result = result.lower()

    if options.get("reverse", False):
        result = result[::-1]

    repeat = options.get("repeat", 1)
    result = result * repeat

    return result


if __name__ == "__main__":
    # Run the MCP server directly using stdio transport
    # The run() method handles the async event loop.
    mcp_server.run(transport="stdio")