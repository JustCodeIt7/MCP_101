"""
# Using MCP with LangChain: Building Smarter Pipelines

This script demonstrates how to integrate MCP (Model-Centric Programming) with LangChain
to build more powerful and flexible AI pipelines. It covers the basics of LangChain,
integration techniques, and practical use cases.

## Prerequisites
- Python 3.7+
- An MCP server running (see Video 3)
- Required packages: langchain, requests, json

## Key Concepts
1. Overview of LangChain
2. Integrating MCP with LangChain
3. Building pipelines with MCP and LangChain
4. Practical use cases for MCP-LangChain integration
"""

import os
import json
import requests
from typing import Dict, Any, List, Optional, Union

# Import LangChain components
from langchain.llms.base import LLM
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent
from langchain.schema import AgentAction, AgentFinish, HumanMessage


# Import our MCP client from Video 4
class MCPClient:
    """
    A client for interacting with MCP servers.

    This class provides methods to initialize a connection to an MCP server
    and perform various operations like sending requests and handling responses.
    """

    def __init__(self, server_url: str, api_key: Optional[str] = None):
        """
        Initialize an MCP client with server URL and optional API key.

        Args:
            server_url: The URL of the MCP server
            api_key: Optional API key for authentication
        """
        self.server_url = server_url
        self.api_key = api_key
        self.headers = {
            'Content-Type': 'application/json'
        }

        if api_key:
            self.headers['Authorization'] = f'Bearer {api_key}'

    def send_request(self, endpoint: str, data: Dict[str, Any], method: str = "POST") -> Dict[str, Any]:
        """
        Send a request to the MCP server.

        Args:
            endpoint: The API endpoint to call
            data: The data to send in the request
            method: HTTP method (GET, POST, PUT, DELETE)

        Returns:
            Dict containing the server response
        """
        url = f"{self.server_url}/{endpoint}"

        try:
            if method.upper() == "GET":
                response = requests.get(url, params=data, headers=self.headers)
            elif method.upper() == "POST":
                response = requests.post(url, json=data, headers=self.headers)
            elif method.upper() == "PUT":
                response = requests.put(url, json=data, headers=self.headers)
            elif method.upper() == "DELETE":
                response = requests.delete(url, json=data, headers=self.headers)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")

            # Check if the request was successful
            response.raise_for_status()

            # Parse and return the JSON response
            return response.json()

        except requests.exceptions.RequestException as e:
            print(f"❌ Request Error: {e}")
            return {"error": str(e)}

    def predict(self, model_name: str, inputs: Dict[str, Any], parameters: Optional[Dict[str, Any]] = None) -> Dict[
        str, Any]:
        """
        Send a prediction request to the MCP server.

        Args:
            model_name: Name of the model to use for prediction
            inputs: Input data for the model
            parameters: Optional parameters for the prediction

        Returns:
            Dict containing the prediction results
        """
        data = {
            "model": model_name,
            "inputs": inputs
        }

        if parameters:
            data["parameters"] = parameters

        return self.send_request("predict", data)


# Part 1: Creating a LangChain-compatible MCP LLM class
class MCPLLM(LLM):
    """
    A LangChain-compatible LLM class that uses MCP for predictions.

    This class allows MCP models to be used seamlessly within LangChain pipelines.
    """

    client: MCPClient
    model_name: str
    model_parameters: Dict[str, Any] = {}

    @property
    def _llm_type(self) -> str:
        """Return the type of LLM."""
        return "mcp"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """
        Call the MCP model with the given prompt.

        Args:
            prompt: The input prompt for the model
            stop: Optional list of stop sequences

        Returns:
            The model's response as a string
        """
        # Prepare parameters with stop sequences if provided
        parameters = self.model_parameters.copy()
        if stop:
            parameters["stop"] = stop

        # Send the request to the MCP server
        response = self.client.predict(
            model_name=self.model_name,
            inputs={"prompt": prompt},
            parameters=parameters
        )

        # Check for errors
        if "error" in response:
            raise ValueError(f"Error from MCP server: {response['error']}")

        # Extract and return the generated text
        return response.get("output", "")

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Return identifying parameters."""
        return {
            "model_name": self.model_name,
            "model_parameters": self.model_parameters
        }


# Part 2: Building a simple LangChain pipeline with MCP
def simple_langchain_pipeline():
    """
    Demonstrates a simple LangChain pipeline using MCP.

    This example shows how to create a basic LLMChain with an MCP model.
    """
    print("\n" + "=" * 50)
    print("Simple LangChain Pipeline with MCP")
    print("=" * 50)

    # Initialize the MCP client
    client = MCPClient(server_url="http://localhost:8000", api_key="your_api_key_here")

    # Create an MCP LLM
    mcp_llm = MCPLLM(
        client=client,
        model_name="text-generation",
        model_parameters={"temperature": 0.7, "max_tokens": 100}
    )

    # Create a prompt template
    template = """
    You are a helpful assistant that provides information about {topic}.
    
    User question: {question}
    
    Your response:
    """

    prompt = PromptTemplate(
        input_variables=["topic", "question"],
        template=template
    )

    # Create an LLMChain
    chain = LLMChain(llm=mcp_llm, prompt=prompt)

    # Run the chain
    response = chain.run(topic="artificial intelligence", question="What is machine learning?")

    print("\nQuestion: What is machine learning?")
    print(f"Response: {response}")

    # Another example
    response = chain.run(topic="climate change", question="How does global warming affect sea levels?")

    print("\nQuestion: How does global warming affect sea levels?")
    print(f"Response: {response}")

    print("\n" + "=" * 50)


# Part 3: Creating a conversational chain with memory
def conversational_chain():
    """
    Demonstrates a conversational chain with memory using MCP.

    This example shows how to create a conversational agent that remembers previous interactions.
    """
    print("\n" + "=" * 50)
    print("Conversational Chain with Memory")
    print("=" * 50)

    # Initialize the MCP client
    client = MCPClient(server_url="http://localhost:8000", api_key="your_api_key_here")

    # Create an MCP LLM
    mcp_llm = MCPLLM(
        client=client,
        model_name="chat-model",
        model_parameters={"temperature": 0.8, "max_tokens": 150}
    )

    # Create a conversation memory
    memory = ConversationBufferMemory(memory_key="chat_history")

    # Create a prompt template for conversation
    template = """
    The following is a friendly conversation between a human and an AI assistant.
    The assistant is helpful, creative, clever, and very friendly.
    
    Chat history:
    {chat_history}
    
    Human: {human_input}
    AI Assistant:
    """

    prompt = PromptTemplate(
        input_variables=["chat_history", "human_input"],
        template=template
    )

    # Create a conversation chain
    conversation = LLMChain(
        llm=mcp_llm,
        prompt=prompt,
        memory=memory,
        verbose=True
    )

    # Simulate a conversation
    print("\nStarting conversation...")

    # First message
    response = conversation.predict(human_input="Hi there! My name is Alice.")
    print(f"AI: {response}")

    # Second message
    response = conversation.predict(human_input="What can you tell me about machine learning?")
    print(f"AI: {response}")

    # Third message (referencing previous context)
    response = conversation.predict(human_input="Can you explain it in simpler terms?")
    print(f"AI: {response}")

    # Fourth message (testing memory)
    response = conversation.predict(human_input="By the way, what's my name?")
    print(f"AI: {response}")

    print("\n" + "=" * 50)


# Part 4: Building a tool-using agent with MCP and LangChain
def tool_using_agent():
    """
    Demonstrates a tool-using agent with MCP and LangChain.

    This example shows how to create an agent that can use tools to accomplish tasks.
    """
    print("\n" + "=" * 50)
    print("Tool-Using Agent with MCP and LangChain")
    print("=" * 50)

    # Initialize the MCP client
    client = MCPClient(server_url="http://localhost:8000", api_key="your_api_key_here")

    # Create an MCP LLM
    mcp_llm = MCPLLM(
        client=client,
        model_name="reasoning-model",
        model_parameters={"temperature": 0.2, "max_tokens": 300}
    )

    # Define some tools for the agent to use
    def get_weather(location: str) -> str:
        """Get the current weather for a location."""
        # In a real application, this would call a weather API
        return f"The weather in {location} is currently sunny with a temperature of 72°F."

    def search_database(query: str) -> str:
        """Search a database for information."""
        # In a real application, this would query a database
        if "population" in query.lower():
            return "The population of the United States is approximately 331 million people."
        elif "capital" in query.lower():
            return "The capital of France is Paris."
        else:
            return "No relevant information found in the database."

    def calculate(expression: str) -> str:
        """Evaluate a mathematical expression."""
        try:
            return str(eval(expression))
        except Exception as e:
            return f"Error evaluating expression: {e}"

    # Create LangChain tools
    tools = [
        Tool(
            name="Weather",
            func=get_weather,
            description="Useful for getting the current weather in a location. Input should be a location name."
        ),
        Tool(
            name="Database",
            func=search_database,
            description="Useful for searching a database for information. Input should be a search query."
        ),
        Tool(
            name="Calculator",
            func=calculate,
            description="Useful for performing mathematical calculations. Input should be a mathematical expression."
        )
    ]

    # Create a prompt template for the agent
    template = """
    You are an AI assistant that can use tools to help answer questions.
    
    You have access to the following tools:
    {tools}
    
    Use the following format:
    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question
    
    Begin!
    
    Question: {input}
    Thought:
    """

    # Create the agent
    from langchain.agents import AgentOutputParser
    from langchain.schema import AgentAction, AgentFinish
    import re

    class CustomOutputParser(AgentOutputParser):
        def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
            # Check if the agent has finished
            if "Final Answer:" in text:
                return AgentFinish(
                    return_values={"output": text.split("Final Answer:")[-1].strip()},
                    log=text
                )

            # Parse the action and action input
            match = re.search(r"Action: (.*?)[\n]Action Input: (.*)", text, re.DOTALL)
            if not match:
                raise ValueError(f"Could not parse LLM output: {text}")

            action = match.group(1).strip()
            action_input = match.group(2).strip()

            return AgentAction(tool=action, tool_input=action_input, log=text)

    # Create the agent executor
    agent = LLMSingleActionAgent(
        llm_chain=LLMChain(llm=mcp_llm, prompt=PromptTemplate.from_template(template)),
        output_parser=CustomOutputParser(),
        stop=["\nObservation:"],
        allowed_tools=[tool.name for tool in tools]
    )

    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        verbose=True
    )

    # Run the agent with some example queries
    print("\nExample 1: Weather query")
    result = agent_executor.run("What's the weather like in New York?")
    print(f"Final answer: {result}")

    print("\nExample 2: Database query")
    result = agent_executor.run("What is the capital of France?")
    print(f"Final answer: {result}")

    print("\nExample 3: Calculation")
    result = agent_executor.run("What is 125 * 37?")
    print(f"Final answer: {result}")

    print("\n" + "=" * 50)


# Part 5: Practical use case - Document Q&A with MCP and LangChain
def document_qa_pipeline():
    """
    Demonstrates a document Q&A pipeline using MCP and LangChain.

    This example shows how to create a pipeline that can answer questions about documents.
    """
    print("\n" + "=" * 50)
    print("Document Q&A Pipeline with MCP and LangChain")
    print("=" * 50)

    # For this example, we'll use a simulated document retrieval system
    # In a real application, you would use LangChain's document loaders and retrievers

    # Sample documents
    documents = [
        "Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to natural intelligence displayed by animals including humans.",
        "Machine learning is a subset of AI that focuses on the development of algorithms that can learn from and make predictions based on data.",
        "Deep learning is a subset of machine learning that uses neural networks with many layers (hence 'deep') to analyze various factors of data.",
        "Natural Language Processing (NLP) is a field of AI that gives machines the ability to read, understand, and derive meaning from human languages.",
        "Computer vision is a field of AI that enables computers to derive meaningful information from digital images, videos, and other visual inputs."
    ]

    # Initialize the MCP client
    client = MCPClient(server_url="http://localhost:8000", api_key="your_api_key_here")

    # Create an MCP LLM
    mcp_llm = MCPLLM(
        client=client,
        model_name="qa-model",
        model_parameters={"temperature": 0.3, "max_tokens": 200}
    )

    # Simulated document retrieval function
    def retrieve_relevant_documents(query: str) -> List[str]:
        """
        Retrieve documents relevant to the query.

        In a real application, this would use a vector database or other retrieval system.
        For this example, we'll use a simple keyword matching approach.
        """
        query_terms = query.lower().split()
        relevant_docs = []

        for doc in documents:
            doc_lower = doc.lower()
            if any(term in doc_lower for term in query_terms):
                relevant_docs.append(doc)

        return relevant_docs if relevant_docs else [documents[0]]  # Return at least one document

    # Create a prompt template for document Q&A
    template = """
    You are an AI assistant that answers questions based on the provided documents.
    
    Documents:
    {documents}
    
    Question: {question}
    
    Your answer should be based only on the information provided in the documents.
    If the documents don't contain the answer, say "I don't have enough information to answer this question."
    
    Answer:
    """

    prompt = PromptTemplate(
        input_variables=["documents", "question"],
        template=template
    )

    # Create the Q&A chain
    qa_chain = LLMChain(llm=mcp_llm, prompt=prompt)

    # Function to answer questions
    def answer_question(question: str) -> str:
        """
        Answer a question based on the documents.

        Args:
            question: The question to answer

        Returns:
            The answer to the question
        """
        # Retrieve relevant documents
        relevant_docs = retrieve_relevant_documents(question)

        # Format the documents as a string
        docs_str = "\n".join([f"- {doc}" for doc in relevant_docs])

        # Run the Q&A chain
        answer = qa_chain.run(documents=docs_str, question=question)

        return answer

    # Example questions
    questions = [
        "What is artificial intelligence?",
        "How is machine learning related to AI?",
        "What is deep learning?",
        "What is the purpose of NLP?",
        "Can AI understand images?"
    ]

    # Answer each question
    for i, question in enumerate(questions, 1):
        print(f"\nQuestion {i}: {question}")
        answer = answer_question(question)
        print(f"Answer: {answer}")

    print("\n" + "=" * 50)


# Main function to run all examples
def main():
    """
    Main function to run all examples.
    """
    print("\n" + "=" * 50)
    print("MCP with LangChain: Building Smarter Pipelines")
    print("=" * 50)

    print("\nThis script demonstrates how to integrate MCP with LangChain to build powerful AI pipelines.")
    print("We'll explore several examples, from simple chains to complex agents.")

    # Run the examples
    simple_langchain_pipeline()
    conversational_chain()
    tool_using_agent()
    document_qa_pipeline()

    print("\n" + "=" * 50)
    print("All examples completed!")
    print("=" * 50)


if __name__ == "__main__":
    main()
