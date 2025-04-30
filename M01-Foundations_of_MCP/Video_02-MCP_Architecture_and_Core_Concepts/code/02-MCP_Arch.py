"""
MCP Architecture and Core Concepts
==================================

This script demonstrates the key architectural components of MCP (Machine Conversation Protocol)
and how they interact with each other. It covers:

1. MCP server-client architecture
2. Context management in MCP
3. Communication protocols in MCP
4. Example of defining a context
5. Basic server-client interaction

Prerequisites:
- Python 3.8+
- mcp-python package
"""

# Import necessary libraries
import asyncio
import json
from typing import Dict, List, Optional, Any

# Note: In a real implementation, you would import from the mcp package
# For demonstration purposes, we'll define simplified versions of MCP components

# =====================================================================
# PART 1: MCP ARCHITECTURE OVERVIEW
# =====================================================================

"""
MCP Architecture consists of three main components:
1. Server: Manages contexts and handles client requests
2. Client: Sends requests to the server and receives responses
3. Context: Stores conversation state and manages the flow of the conversation

The communication between these components follows a specific protocol
that ensures consistent and reliable message exchange.
"""


# =====================================================================
# PART 2: CONTEXT MANAGEMENT
# =====================================================================

class MCPContext:
    """
    A simplified version of an MCP Context.

    In MCP, a context represents a conversation state and includes:
    - Messages: The history of the conversation
    - Metadata: Additional information about the conversation
    - Parameters: Configuration for the conversation
    """

    def __init__(self, context_id: str, parameters: Optional[Dict[str, Any]] = None):
        self.context_id = context_id
        self.messages: List[Dict[str, Any]] = []
        self.metadata: Dict[str, Any] = {}
        self.parameters = parameters or {}

    def add_message(self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None):
        """Add a message to the context."""
        message = {
            "role": role,
            "content": content
        }
        if metadata:
            message["metadata"] = metadata

        self.messages.append(message)
        return message

    def get_messages(self) -> List[Dict[str, Any]]:
        """Get all messages in the context."""
        return self.messages

    def to_dict(self) -> Dict[str, Any]:
        """Convert the context to a dictionary for serialization."""
        return {
            "context_id": self.context_id,
            "messages": self.messages,
            "metadata": self.metadata,
            "parameters": self.parameters
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MCPContext':
        """Create a context from a dictionary."""
        context = cls(data["context_id"], data.get("parameters"))
        context.messages = data.get("messages", [])
        context.metadata = data.get("metadata", {})
        return context


# =====================================================================
# PART 3: SERVER IMPLEMENTATION
# =====================================================================

class MCPServer:
    """
    A simplified version of an MCP Server.

    The server is responsible for:
    - Managing contexts
    - Processing client requests
    - Maintaining the state of conversations
    """

    def __init__(self):
        self.contexts: Dict[str, MCPContext] = {}

    def create_context(self, context_id: str, parameters: Optional[Dict[str, Any]] = None) -> MCPContext:
        """Create a new context."""
        context = MCPContext(context_id, parameters)
        self.contexts[context_id] = context
        return context

    def get_context(self, context_id: str) -> Optional[MCPContext]:
        """Get a context by ID."""
        return self.contexts.get(context_id)

    async def process_message(self, context_id: str, message: Dict[str, Any]) -> Dict[str, Any]:
        """Process a message from a client."""
        # Get or create the context
        context = self.get_context(context_id)
        if not context:
            context = self.create_context(context_id)

        # Add the message to the context
        context.add_message(message["role"], message["content"], message.get("metadata"))

        # In a real implementation, this would process the message using an LLM or other service
        # For demonstration, we'll just echo the message with a prefix
        response = {
            "role": "assistant",
            "content": f"Processed: {message['content']}"
        }

        # Add the response to the context
        context.add_message(response["role"], response["content"])

        return response


# =====================================================================
# PART 4: CLIENT IMPLEMENTATION
# =====================================================================

class MCPClient:
    """
    A simplified version of an MCP Client.

    The client is responsible for:
    - Sending requests to the server
    - Receiving responses from the server
    - Managing the client-side state of the conversation
    """

    def __init__(self, server: MCPServer):
        self.server = server
        self.current_context_id: Optional[str] = None

    def set_context(self, context_id: str):
        """Set the current context ID."""
        self.current_context_id = context_id

    async def send_message(self, content: str, role: str = "user", metadata: Optional[Dict[str, Any]] = None) -> Dict[
        str, Any]:
        """Send a message to the server."""
        if not self.current_context_id:
            raise ValueError("No context ID set. Call set_context() first.")

        message = {
            "role": role,
            "content": content
        }
        if metadata:
            message["metadata"] = metadata

        response = await self.server.process_message(self.current_context_id, message)
        return response


# =====================================================================
# PART 5: COMMUNICATION PROTOCOLS
# =====================================================================

"""
MCP supports multiple communication protocols:
1. HTTP/WebSocket: For web-based applications
2. gRPC: For high-performance applications
3. Standard I/O: For command-line applications
4. Server-Sent Events (SSE): For streaming responses

Each protocol has its own advantages and use cases.
"""


# Example of a simple HTTP-like protocol (simplified for demonstration)
async def http_request(server: MCPServer, request: Dict[str, Any]) -> Dict[str, Any]:
    """Simulate an HTTP request to the MCP server."""
    context_id = request.get("context_id")
    message = request.get("message")

    if not context_id or not message:
        return {"error": "Invalid request"}

    response = await server.process_message(context_id, message)

    return {
        "context_id": context_id,
        "response": response
    }


# =====================================================================
# PART 6: DEMONSTRATION
# =====================================================================

async def main():
    """Demonstrate the MCP architecture and core concepts."""
    print("MCP Architecture and Core Concepts Demonstration")
    print("===============================================\n")

    # Create a server
    server = MCPServer()
    print("1. Created MCP Server")

    # Create a client
    client = MCPClient(server)
    print("2. Created MCP Client")

    # Set up a context
    context_id = "demo-context"
    client.set_context(context_id)
    print(f"3. Set client context to: {context_id}")

    # Create the context on the server with parameters
    parameters = {
        "model": "gpt-4",
        "temperature": 0.7,
        "max_tokens": 150
    }
    server.create_context(context_id, parameters)
    print(f"4. Created context on server with parameters: {parameters}")

    # Send a message from the client to the server
    user_message = "Hello, I'd like to learn about MCP!"
    print(f"\n5. Sending message: '{user_message}'")
    response = await client.send_message(user_message)
    print(f"6. Received response: '{response['content']}'")

    # Send another message
    user_message = "What are the key components of MCP?"
    print(f"\n7. Sending message: '{user_message}'")
    response = await client.send_message(user_message)
    print(f"8. Received response: '{response['content']}'")

    # Get the context and display its contents
    context = server.get_context(context_id)
    print("\n9. Current context state:")
    print(json.dumps(context.to_dict(), indent=2))

    # Demonstrate HTTP-like protocol
    print("\n10. Demonstrating HTTP-like protocol:")
    http_response = await http_request(server, {
        "context_id": context_id,
        "message": {
            "role": "user",
            "content": "Can you explain context management in MCP?"
        }
    })
    print(f"11. HTTP Response: {http_response['response']['content']}")

    print("\nDemonstration complete!")


if __name__ == "__main__":
    asyncio.run(main())
