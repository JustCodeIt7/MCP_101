from mcp import ClientSession, stdio_client
import json
import subprocess
import sys

def pretty_print_json(data):
    """Pretty print JSON data."""
    print(json.dumps(data, indent=2))

def main():
    # Path to the server script - adjust if needed
    server_path = "../02-User_Profile_MCP_Server/server.py"
    
    # Spawn the server in a subprocess
    # Note: You can also run the server separately and comment out this part
    server_process = subprocess.Popen(
        [sys.executable, server_path],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )
    
    # Connect to the server
    with ClientSession(stdio_client.spawn(f"python {server_path}")) as session:
        # List available tools
        print("Available tools:")
        tools = session.list_tools()
        pretty_print_json(tools)
        
        # Call the add tool
        print("\nCalling add tool with a=2, b=3:")
        result = session.call_tool("add", a=2, b=3)
        print(f"Result: {result}")
        
        # Read user profile resource
        print("\nReading user profile for 'alice':")
        profile = session.read_resource("users://alice/profile")
        pretty_print_json(profile)
        
        # Try with a non-existent user
        print("\nReading user profile for 'nonexistent':")
        profile = session.read_resource("users://nonexistent/profile")
        pretty_print_json(profile)

if __name__ == "__main__":
    main()