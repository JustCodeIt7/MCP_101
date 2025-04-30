from mcp.server.fastmcp import FastMCP

# Instantiate FastMCP
mcp = FastMCP("search_server")

@mcp.tool()
def web_search(query: str) -> str:
    """
    Perform a web search for the given query.
    
    This is a dummy implementation that returns predefined results based on keywords in the query.
    In a real implementation, this would connect to a search API.
    
    Args:
        query: The search query string
        
    Returns:
        Search results as a string
    """
    # Dummy search results based on keywords
    if "mcp" in query.lower():
        return """
        Search results for "MCP":
        
        1. Model-Client-Proxy (MCP) - A framework for AI tool integration
           MCP is a protocol for connecting AI models with external tools and resources.
           It provides a standardized way for models to request information or actions.
        
        2. GitHub: mcp-python - Python implementation of the MCP protocol
           A Python library that implements the MCP protocol for both clients and servers.
           Supports various transports including stdio and HTTP/SSE.
        
        3. MCP vs Function Calling: What's the difference?
           MCP provides a more flexible and extensible approach compared to function calling.
           It supports resources, streaming, and more complex interactions.
        """
    elif "python" in query.lower():
        return """
        Search results for "Python":
        
        1. Python.org - Official website of the Python programming language
           Download the latest version, access documentation, and join the community.
        
        2. Python Tutorial - W3Schools
           Learn Python with our easy to follow tutorials and examples.
        
        3. The Python Standard Library
           Documentation for the standard library that's distributed with Python.
        """
    else:
        return f"""
        Search results for "{query}":
        
        1. Wikipedia: {query}
           General information about {query} from the free encyclopedia.
        
        2. Latest news about {query}
           Recent articles and updates related to {query}.
        
        3. {query} - Official resources
           Main sources of information about {query}.
        """

if __name__ == "__main__":
    mcp.run(transport="stdio")