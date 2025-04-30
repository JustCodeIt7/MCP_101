from mcp.server.fastmcp import FastMCP

# Instantiate FastMCP
mcp = FastMCP("file_server")

@mcp.tool()
def read_file(path: str) -> str:
    """
    Read the content of a file at the given path.
    
    This is a dummy implementation that returns predefined content based on the file path.
    In a real implementation, this would read from the actual file system with proper security checks.
    
    Args:
        path: The path to the file to read
        
    Returns:
        The content of the file as a string
    """
    # Dummy file contents based on path
    if path.lower() == "readme.md":
        return """
        # MCP Project
        
        ## Overview
        This project demonstrates the Model-Client-Proxy (MCP) protocol for AI tool integration.
        
        ## Features
        - Standardized communication between AI models and tools
        - Support for various transports (stdio, HTTP/SSE)
        - Resource endpoints for data access
        - Tool endpoints for actions and computations
        
        ## Getting Started
        1. Install the MCP library: `pip install mcp`
        2. Create a server with tools and resources
        3. Connect a client to the server
        4. Start building AI-powered applications!
        """
    elif path.lower().endswith(".py"):
        return """
        # Example Python code
        
        def hello_world():
            \"\"\"Print a greeting message.\"\"\"
            print("Hello, world!")
            
        class Calculator:
            \"\"\"A simple calculator class.\"\"\"
            
            def add(self, a, b):
                \"\"\"Add two numbers.\"\"\"
                return a + b
                
            def subtract(self, a, b):
                \"\"\"Subtract b from a.\"\"\"
                return a - b
                
        if __name__ == "__main__":
            hello_world()
            calc = Calculator()
            print(f"2 + 3 = {calc.add(2, 3)}")
        """
    else:
        return f"""
        This is dummy content for the file: {path}
        
        In a real implementation, this would be the actual content of the file
        at the specified path, with appropriate security checks to prevent
        unauthorized access to sensitive files.
        
        For demonstration purposes, this is just placeholder text.
        """

if __name__ == "__main__":
    mcp.run(transport="stdio")