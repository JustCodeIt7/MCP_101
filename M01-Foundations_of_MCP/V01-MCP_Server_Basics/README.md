# MCP Server Demo

This repository contains a simple demonstration of an MCP (Machine Conversation Protocol) server built using the Python MCP SDK. The server exposes several Python functions as MCP Tools that can be used by LLM clients.

## Prerequisites

- Python 3.8 or higher
- MCP Python SDK

## Installation

1. Install the MCP Python SDK:

```bash
pip install mcp
```

2. Clone this repository or download the `mcp_server.py` file.

## Running the Server

Run the MCP server using:

```bash
python mcp_server.py
```

This will start the server and make it available via the stdio transport, making it usable as a subprocess by MCP clients.

## Available Tools

The server exposes the following tools:

### add

Adds two numbers together.

Example:
```
Input: a=2, b=3
Output: 5
```

### multiply

Multiplies two numbers together.

Example:
```
Input: a=4, b=5
Output: 20
```

### search_books

Searches for books matching the given query.

Example:
```
Input: query="python", max_results=2
Output: [
  {"title": "Python Crash Course", "author": "Eric Matthes", "year": 2019},
  {"title": "Fluent Python", "author": "Luciano Ramalho", "year": 2021}
]
```

### format_text

Formats text according to the specified options.

Example:
```
Input: text="Hello, World!", options={"uppercase": true, "repeat": 2}
Output: "HELLO, WORLD!HELLO, WORLD!"
```

## How It Works

The MCP server is created using the `FastMCP` class from the MCP Python SDK. Each function is decorated with `@mcp_server.tool()` to expose it as an MCP Tool.

The server runs using the stdio transport, which allows it to communicate via standard input and output. This makes it easy to run the server as a subprocess from an MCP client.

## Important Concepts

- **FastMCP**: A high-level class for creating MCP servers that abstracts away the underlying protocol complexity.
- **@mcp_server.tool() Decorator**: Transforms a Python function into an MCP Tool discoverable by clients.
- **Type Hints**: Used to define the input arguments and return types of each tool.
- **Docstrings**: Provide natural language descriptions of each tool's purpose, crucial for LLMs to understand how to use the tools.
- **Stdio Transport**: A transport mechanism for running MCP servers locally as subprocesses.

## Learn More

For more information on the MCP Protocol and the Python SDK, check out the official documentation.