import subprocess
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_mcp_adapters import MCPToolAdapter

# Start the MCP server as a subprocess
server_path = "M04-Example_MCP_Servers/01-Basic_Calculator_MCP_Server/server.py"
server_process = subprocess.Popen(
    ["python", server_path],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True
)

# Initialize Ollama LLM
llm = ChatOllama(model="llama2")

# Connect to the MCP server and load tools
adapter = MCPToolAdapter(endpoint="stdio", process=server_process)
tools = adapter.get_tools()

# Create a simple prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that can perform calculations using external tools."),
    ("human", "{input}")
])

# Create the chain
chain = (
        prompt
        | llm.bind_tools(tools)
        | StrOutputParser()
)

# Test the add tool
try:
    result = chain.invoke({"input": "I need to add 42 and 13. Can you help me?"})
    print("Result:", result)

    # Test another calculation
    result = chain.invoke({"input": "What's 75 plus 25?"})
    print("Result:", result)

    # Test the echo tool
    result = chain.invoke({"input": "Echo this message: Hello, MCP server!"})
    print("Result:", result)

except Exception as e:
    print(f"Error: {e}")
finally:
    # Clean up the server process
    server_process.terminate()
    print("Server process terminated")