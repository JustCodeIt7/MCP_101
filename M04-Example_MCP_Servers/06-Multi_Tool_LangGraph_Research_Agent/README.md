# Multi-Tool LangGraph Research Agent

This project demonstrates how to create a research agent that uses multiple MCP servers and LangGraph to orchestrate a workflow for answering questions.

## Overview

The agent can:
1. Search for information using a web search tool
2. Read files for additional context
3. Plan a research strategy
4. Synthesize information into a comprehensive answer

## Components

### MCP Servers

- **Search Server**: Provides a `web_search` tool that returns search results for a query
- **File Server**: Provides a `read_file` tool that returns the content of a file

### LangGraph Workflow

The workflow is defined as a state machine with the following nodes:
- **Planner**: Decides what action to take next based on the question and gathered information
- **Search Web**: Executes a web search using the search server
- **Read File**: Reads a file using the file server
- **Synthesize**: Combines all gathered information to generate a final answer

## How It Works

1. The agent starts by creating a plan to answer the question
2. Based on the plan, it decides whether to search the web or read a file
3. After gathering information, it returns to the planner to decide the next step
4. Once enough information is gathered, it synthesizes a final answer
5. The process continues until the task is complete

## Usage

To run the agent:

```bash
python graph_agent.py
```

This will:
1. Start both MCP servers as subprocesses
2. Create the LangGraph workflow
3. Run the agent with a sample question
4. Clean up the server processes when done

## Requirements

- Python 3.9+
- mcp
- fastmcp
- langgraph
- langchain
- langchain-openai
- langchain-mcp-adapters

## Example

The default example asks the agent to "Search for MCP intro and read README.md". The agent will:
1. Search for information about MCP
2. Read the README.md file
3. Combine this information to provide a comprehensive answer about MCP