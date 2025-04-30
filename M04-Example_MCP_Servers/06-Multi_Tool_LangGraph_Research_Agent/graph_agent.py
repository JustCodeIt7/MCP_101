import subprocess
import sys
from typing import TypedDict, Annotated, Literal, List
import os
from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters import MultiServerMCPClient
from langgraph.graph import StateGraph, END

# Define the workflow state
class WorkflowState(TypedDict):
    question: str
    plan: str
    intermediate: List[str]
    answer: str
    next_action: Annotated[str, Literal["search", "read", "synthesize", "end"]]

def create_research_agent():
    # Get the current directory to find server scripts
    current_dir = Path(__file__).parent
    search_server_path = current_dir / "servers" / "search_server.py"
    file_server_path = current_dir / "servers" / "file_server.py"
    
    # Start the server processes
    search_server_process = subprocess.Popen(
        [sys.executable, str(search_server_path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )
    
    file_server_process = subprocess.Popen(
        [sys.executable, str(file_server_path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )
    
    # Create the MCP client that connects to both servers
    client = MultiServerMCPClient([
        f"stdio:python {search_server_path}",
        f"stdio:python {file_server_path}"
    ])
    
    # Load tools from the servers
    tools = client.load_tools()
    
    # Create an LLM
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    
    # Define the nodes for the graph
    
    # 1. Planner node - decides what to do next
    planner_prompt = ChatPromptTemplate.from_template("""
    You are a research assistant that helps answer questions.
    
    Based on the question, decide what action to take next:
    - "search": If you need to search the web for information
    - "read": If you need to read a specific file
    - "synthesize": If you have enough information to answer the question
    - "end": If the task is complete
    
    Question: {question}
    Current plan: {plan}
    Information gathered so far: {intermediate}
    
    Decide the next action and explain your reasoning.
    """)
    
    def planner(state: WorkflowState) -> WorkflowState:
        """Decide what action to take next."""
        if not state.get("plan"):
            # First time planning, create an initial plan
            planning_prompt = ChatPromptTemplate.from_template("""
            You are a research assistant that helps answer questions.
            
            Create a step-by-step plan to answer this question:
            {question}
            
            Your plan should include specific steps like searching for information
            or reading specific files.
            """)
            response = llm.invoke(planning_prompt.format(question=state["question"]))
            state["plan"] = response.content
        
        # Decide next action
        response = llm.invoke(planner_prompt.format(**state))
        
        # Extract the next action from the response
        content = response.content.lower()
        if "search" in content and ("web" in content or "internet" in content):
            state["next_action"] = "search"
        elif "read" in content and ("file" in content or "document" in content):
            state["next_action"] = "read"
        elif "synthesize" in content or "answer" in content:
            state["next_action"] = "synthesize"
        elif "end" in content or "complete" in content:
            state["next_action"] = "end"
        else:
            # Default to search if unclear
            state["next_action"] = "search"
            
        return state
    
    # 2. Tool execution nodes
    def search_web(state: WorkflowState) -> WorkflowState:
        """Execute web search based on the question."""
        search_tool = next(tool for tool in tools if tool.name == "web_search")
        result = search_tool.invoke({"query": state["question"]})
        
        if not state.get("intermediate"):
            state["intermediate"] = []
        
        state["intermediate"].append(f"Search results: {result}")
        return state
    
    def read_file(state: WorkflowState) -> WorkflowState:
        """Read a file based on the question or plan."""
        # Extract filename from question or plan
        # This is a simple heuristic - in a real system, you'd use NLP to extract the filename
        file_tool = next(tool for tool in tools if tool.name == "read_file")
        
        # Default to README.md if no specific file is mentioned
        filename = "README.md"
        
        # Check if a specific file is mentioned in the question or plan
        for text in [state["question"], state["plan"]]:
            if "readme" in text.lower():
                filename = "README.md"
                break
            elif ".py" in text.lower():
                filename = "example.py"
                break
        
        result = file_tool.invoke({"path": filename})
        
        if not state.get("intermediate"):
            state["intermediate"] = []
        
        state["intermediate"].append(f"Content of {filename}: {result}")
        return state
    
    # 3. Synthesize node - generate the final answer
    synthesize_prompt = ChatPromptTemplate.from_template("""
    You are a research assistant that helps answer questions.
    
    Based on the information gathered, provide a comprehensive answer to the question.
    
    Question: {question}
    Information gathered:
    {intermediate}
    
    Your answer:
    """)
    
    def synthesize(state: WorkflowState) -> WorkflowState:
        """Generate the final answer based on gathered information."""
        intermediate_text = "\n".join(state["intermediate"])
        response = llm.invoke(synthesize_prompt.format(
            question=state["question"],
            intermediate=intermediate_text
        ))
        
        state["answer"] = response.content
        state["next_action"] = "end"
        return state
    
    # Create the graph
    workflow = StateGraph(WorkflowState)
    
    # Add nodes
    workflow.add_node("planner", planner)
    workflow.add_node("search_web", search_web)
    workflow.add_node("read_file", read_file)
    workflow.add_node("synthesize", synthesize)
    
    # Add edges
    workflow.add_edge("planner", "search_web", condition=lambda state: state["next_action"] == "search")
    workflow.add_edge("planner", "read_file", condition=lambda state: state["next_action"] == "read")
    workflow.add_edge("planner", "synthesize", condition=lambda state: state["next_action"] == "synthesize")
    workflow.add_edge("planner", END, condition=lambda state: state["next_action"] == "end")
    
    workflow.add_edge("search_web", "planner")
    workflow.add_edge("read_file", "planner")
    workflow.add_edge("synthesize", END)
    
    # Set the entry point
    workflow.set_entry_point("planner")
    
    # Compile the graph
    app = workflow.compile()
    
    return app, search_server_process, file_server_process

def main():
    # Create the research agent
    app, search_process, file_process = create_research_agent()
    
    try:
        # Run the agent with a sample question
        question = "Search for MCP intro and read README.md"
        result = app.invoke({
            "question": question,
            "plan": "",
            "intermediate": [],
            "answer": "",
            "next_action": ""
        })
        
        print("\n=== Question ===")
        print(question)
        
        print("\n=== Plan ===")
        print(result["plan"])
        
        print("\n=== Answer ===")
        print(result["answer"])
        
    finally:
        # Clean up the server processes
        print("\nShutting down servers...")
        search_process.terminate()
        file_process.terminate()

if __name__ == "__main__":
    main()