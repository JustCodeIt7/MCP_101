import subprocess
import sys
from pathlib import Path
import os

from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from langchain_mcp_adapters import MultiServerMCPClient

def main():
    # Get the path to the calculator server
    current_dir = Path(__file__).parent.parent
    calculator_server_path = current_dir / "01-Basic_Calculator_MCP_Server" / "server.py"
    
    print(f"Looking for calculator server at: {calculator_server_path}")
    
    if not calculator_server_path.exists():
        print(f"Error: Calculator server not found at {calculator_server_path}")
        print("Please ensure the Basic Calculator MCP Server is set up correctly.")
        return
    
    # Start the calculator server as a subprocess
    print("Starting calculator server...")
    server_process = subprocess.Popen(
        [sys.executable, str(calculator_server_path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )
    
    try:
        # Create the MCP client that connects to the calculator server
        client = MultiServerMCPClient([f"stdio:python {calculator_server_path}"])
        
        # Load tools from the server
        tools = client.load_tools()
        
        print(f"Loaded {len(tools)} tools from the calculator server:")
        for tool in tools:
            print(f"- {tool.name}: {tool.description}")
        
        # Create an LLM
        llm = ChatOpenAI(model="gpt-4o", temperature=0)
        
        # Create a prompt for the agent
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful assistant that can perform calculations using external tools.
            
When you need to perform a calculation, use the appropriate tool.
Always show your work and explain your reasoning step by step.
"""),
            ("human", "{input}")
        ])
        
        # Create the agent
        agent = create_react_agent(llm, tools, prompt)
        
        # Create the agent executor
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
        
        # Run the agent with a sample question
        print("\n" + "="*50)
        print("Running agent with question: What is 15 plus 27?")
        print("="*50 + "\n")
        
        result = agent_executor.invoke({"input": "What is 15 plus 27?"})
        
        print("\n" + "="*50)
        print("Agent Response:")
        print(result["output"])
        print("="*50 + "\n")
        
        # Try another question
        print("\n" + "="*50)
        print("Running agent with question: If I have 123 apples and give away 45, how many do I have left?")
        print("="*50 + "\n")
        
        result = agent_executor.invoke({
            "input": "If I have 123 apples and give away 45, how many do I have left?"
        })
        
        print("\n" + "="*50)
        print("Agent Response:")
        print(result["output"])
        print("="*50 + "\n")
        
    finally:
        # Clean up the server process
        print("Shutting down calculator server...")
        server_process.terminate()

if __name__ == "__main__":
    main()