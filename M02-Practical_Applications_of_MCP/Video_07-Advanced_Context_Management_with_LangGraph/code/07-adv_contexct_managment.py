"""
# LangGraph and MCP: Advanced Context Management

This script demonstrates how to use LangGraph for advanced context management
with MCP (Model-Centric Programming). It covers the basics of LangGraph,
advanced context management techniques, and real-world examples.

## Prerequisites
- Python 3.7+
- An MCP server running (see Video 3)
- Required packages: langgraph, langchain, requests, json

## Key Concepts
1. Overview of LangGraph
2. Advanced context management techniques
3. Integration of LangGraph with MCP
4. Building projects with LangGraph and MCP
"""

import os
import json
import requests
from typing import Dict, Any, List, Optional, Union, TypedDict, Annotated, Literal
from enum import Enum

# Import LangChain components
from langchain.llms.base import LLM
from langchain.prompts import PromptTemplate
from langchain.schema import AgentAction, AgentFinish, HumanMessage, AIMessage

# Import LangGraph components
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint import MemorySaver


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


# Import our LangChain-compatible MCP LLM class from Video 6
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


# Part 1: Introduction to LangGraph
def introduction_to_langgraph():
    """
    Provides an introduction to LangGraph and its key concepts.

    LangGraph is a framework for building stateful, multi-actor applications with LLMs.
    It extends LangChain by adding support for cyclic graphs and persistent state.
    """
    print("\n" + "=" * 50)
    print("Introduction to LangGraph")
    print("=" * 50)

    print('''
LangGraph is a framework for building stateful, multi-actor applications with LLMs.
It extends LangChain by adding several key features:

1. Cyclic Graphs: Unlike traditional pipelines, LangGraph allows for cycles in the
   execution flow, enabling iterative reasoning and refinement.

2. State Management: LangGraph provides robust state management capabilities,
   allowing applications to maintain context across multiple interactions.

3. Multi-Actor Systems: LangGraph makes it easy to build systems with multiple
   agents that can collaborate to solve complex tasks.

4. Checkpointing: LangGraph supports checkpointing, which allows you to save and
   restore the state of your application at any point.

In this tutorial, we'll explore how to use LangGraph with MCP to build advanced
context-aware applications.
    ''')

    print("\n" + "=" * 50)


# Part 2: Basic LangGraph Example with MCP
class BasicGraphState(TypedDict):
    """Type for the state in our basic graph."""
    question: str
    context: List[str]
    answer: Optional[str]


def basic_langgraph_example():
    """
    Demonstrates a basic LangGraph example using MCP.

    This example shows how to create a simple graph with LangGraph and MCP.
    """
    print("\n" + "=" * 50)
    print("Basic LangGraph Example with MCP")
    print("=" * 50)

    # Initialize the MCP client
    client = MCPClient(server_url="http://localhost:8000", api_key="your_api_key_here")

    # Create an MCP LLM
    mcp_llm = MCPLLM(
        client=client,
        model_name="reasoning-model",
        model_parameters={"temperature": 0.2, "max_tokens": 300}
    )

    # Define the nodes in our graph

    # 1. Context retrieval node
    def retrieve_context(state: BasicGraphState) -> BasicGraphState:
        """Retrieve relevant context for the question."""
        question = state["question"]
        print(f"Retrieving context for: {question}")

        # In a real application, this would query a database or knowledge base
        # For this example, we'll use some hardcoded contexts
        context_database = {
            "python": [
                "Python is a high-level, interpreted programming language.",
                "Python was created by Guido van Rossum and first released in 1991.",
                "Python's design philosophy emphasizes code readability with its notable use of significant whitespace."
            ],
            "machine learning": [
                "Machine learning is a field of study that gives computers the ability to learn without being explicitly programmed.",
                "Machine learning algorithms build a model based on sample data, known as training data, in order to make predictions or decisions.",
                "Common machine learning algorithms include linear regression, decision trees, and neural networks."
            ],
            "climate change": [
                "Climate change refers to long-term shifts in temperatures and weather patterns.",
                "Human activities have been the main driver of climate change since the 1800s, primarily due to burning fossil fuels.",
                "The effects of climate change include rising sea levels, more frequent extreme weather events, and disruptions to ecosystems."
            ]
        }

        # Simple keyword matching to find relevant context
        context = []
        for topic, topic_contexts in context_database.items():
            if topic.lower() in question.lower():
                context.extend(topic_contexts)

        # If no specific context is found, provide a general response
        if not context:
            context = [
                "I don't have specific information on that topic, but I'll try to help based on my general knowledge."]

        # Update the state with the retrieved context
        return {**state, "context": context}

    # 2. Answer generation node
    def generate_answer(state: BasicGraphState) -> BasicGraphState:
        """Generate an answer based on the question and context."""
        question = state["question"]
        context = state["context"]

        print(f"Generating answer for: {question}")
        print(f"Using context: {context}")

        # Create a prompt for the model
        prompt = f"""
        Based on the following context, please answer the question.

        Context:
        {' '.join(context)}

        Question: {question}

        Answer:
        """

        # Generate the answer using the MCP LLM
        answer = mcp_llm(prompt)

        # Update the state with the generated answer
        return {**state, "answer": answer}

    # Create the graph
    workflow = StateGraph(BasicGraphState)

    # Add nodes to the graph
    workflow.add_node("retrieve_context", retrieve_context)
    workflow.add_node("generate_answer", generate_answer)

    # Define the edges in the graph
    workflow.set_entry_point("retrieve_context")
    workflow.add_edge("retrieve_context", "generate_answer")
    workflow.add_edge("generate_answer", END)

    # Compile the graph
    graph = workflow.compile()

    # Run the graph with some example questions
    questions = [
        "What is Python programming language?",
        "Can you explain machine learning in simple terms?",
        "How does climate change affect the environment?"
    ]

    for i, question in enumerate(questions, 1):
        print(f"\nExample {i}: {question}")
        result = graph.invoke({"question": question, "context": [], "answer": None})
        print(f"Answer: {result['answer']}")

    print("\n" + "=" * 50)


# Part 3: Advanced Context Management with LangGraph
class ContextType(str, Enum):
    """Enum for different types of context."""
    USER_PROFILE = "user_profile"
    CONVERSATION_HISTORY = "conversation_history"
    KNOWLEDGE_BASE = "knowledge_base"
    SYSTEM_STATE = "system_state"


class AdvancedGraphState(TypedDict):
    """Type for the state in our advanced graph."""
    user_id: str
    query: str
    contexts: Dict[ContextType, Any]
    current_step: str
    response: Optional[str]
    messages: List[Union[HumanMessage, AIMessage]]


def advanced_context_management():
    """
    Demonstrates advanced context management with LangGraph and MCP.

    This example shows how to manage different types of context in a LangGraph application.
    """
    print("\n" + "=" * 50)
    print("Advanced Context Management with LangGraph")
    print("=" * 50)

    # Initialize the MCP client
    client = MCPClient(server_url="http://localhost:8000", api_key="your_api_key_here")

    # Create an MCP LLM
    mcp_llm = MCPLLM(
        client=client,
        model_name="context-aware-model",
        model_parameters={"temperature": 0.3, "max_tokens": 500}
    )

    # Define the nodes in our graph

    # 1. User profile retrieval node
    def retrieve_user_profile(state: AdvancedGraphState) -> AdvancedGraphState:
        """Retrieve the user profile based on the user ID."""
        user_id = state["user_id"]
        print(f"Retrieving user profile for user: {user_id}")

        # In a real application, this would query a user database
        # For this example, we'll use some hardcoded user profiles
        user_profiles = {
            "user1": {
                "name": "Alice",
                "age": 32,
                "interests": ["machine learning", "hiking", "photography"],
                "expertise_level": "beginner",
                "preferred_language": "English",
                "location": "San Francisco, CA"
            },
            "user2": {
                "name": "Bob",
                "age": 45,
                "interests": ["data science", "gardening", "cooking"],
                "expertise_level": "intermediate",
                "preferred_language": "English",
                "location": "New York, NY"
            },
            "user3": {
                "name": "Charlie",
                "age": 28,
                "interests": ["artificial intelligence", "gaming", "music"],
                "expertise_level": "advanced",
                "preferred_language": "English",
                "location": "London, UK"
            }
        }

        # Get the user profile or a default if not found
        user_profile = user_profiles.get(user_id, {
            "name": "Unknown User",
            "expertise_level": "unknown",
            "interests": []
        })

        # Update the contexts in the state
        contexts = state["contexts"].copy()
        contexts[ContextType.USER_PROFILE] = user_profile

        return {**state, "contexts": contexts, "current_step": "retrieve_conversation_history"}

    # 2. Conversation history retrieval node
    def retrieve_conversation_history(state: AdvancedGraphState) -> AdvancedGraphState:
        """Retrieve the conversation history for the user."""
        user_id = state["user_id"]
        print(f"Retrieving conversation history for user: {user_id}")

        # In a real application, this would query a database of conversation histories
        # For this example, we'll use the messages already in the state
        messages = state["messages"]

        # Update the contexts in the state
        contexts = state["contexts"].copy()
        contexts[ContextType.CONVERSATION_HISTORY] = messages

        return {**state, "contexts": contexts, "current_step": "retrieve_knowledge"}

    # 3. Knowledge retrieval node
    def retrieve_knowledge(state: AdvancedGraphState) -> AdvancedGraphState:
        """Retrieve relevant knowledge based on the query."""
        query = state["query"]
        print(f"Retrieving knowledge for query: {query}")

        # In a real application, this would query a knowledge base or vector database
        # For this example, we'll use some hardcoded knowledge
        knowledge_base = {
            "machine learning": [
                "Machine learning is a field of study that gives computers the ability to learn without being explicitly programmed.",
                "Supervised learning involves training a model on labeled data.",
                "Unsupervised learning involves finding patterns in unlabeled data.",
                "Reinforcement learning involves training agents to take actions in an environment to maximize rewards."
            ],
            "artificial intelligence": [
                "Artificial intelligence (AI) is intelligence demonstrated by machines.",
                "AI can be categorized as either narrow (focused on specific tasks) or general (capable of performing any intellectual task).",
                "Modern AI techniques include machine learning, deep learning, and neural networks.",
                "AI ethics is concerned with the moral implications of creating artificial intelligence."
            ],
            "data science": [
                "Data science is an interdisciplinary field that uses scientific methods, processes, algorithms, and systems to extract knowledge from data.",
                "The data science process typically includes data collection, cleaning, analysis, and visualization.",
                "Common tools in data science include Python, R, SQL, and various machine learning libraries.",
                "Big data refers to datasets that are too large or complex to be dealt with by traditional data-processing software."
            ]
        }

        # Simple keyword matching to find relevant knowledge
        relevant_knowledge = []
        for topic, topic_knowledge in knowledge_base.items():
            if topic.lower() in query.lower():
                relevant_knowledge.extend(topic_knowledge)

        # If no specific knowledge is found, provide a general response
        if not relevant_knowledge:
            relevant_knowledge = ["No specific knowledge found for this query."]

        # Update the contexts in the state
        contexts = state["contexts"].copy()
        contexts[ContextType.KNOWLEDGE_BASE] = relevant_knowledge

        return {**state, "contexts": contexts, "current_step": "generate_response"}

    # 4. Response generation node
    def generate_response(state: AdvancedGraphState) -> AdvancedGraphState:
        """Generate a response based on all available contexts."""
        query = state["query"]
        contexts = state["contexts"]

        print(f"Generating response for query: {query}")

        # Extract contexts
        user_profile = contexts.get(ContextType.USER_PROFILE, {})
        conversation_history = contexts.get(ContextType.CONVERSATION_HISTORY, [])
        knowledge = contexts.get(ContextType.KNOWLEDGE_BASE, [])

        # Format conversation history
        conversation_str = ""
        for message in conversation_history:
            if isinstance(message, HumanMessage):
                conversation_str += f"Human: {message.content}\n"
            elif isinstance(message, AIMessage):
                conversation_str += f"AI: {message.content}\n"

        # Create a prompt for the model
        prompt = f"""
        You are an AI assistant that provides personalized responses based on the user's profile,
        conversation history, and relevant knowledge.

        User Profile:
        - Name: {user_profile.get('name', 'Unknown')}
        - Expertise Level: {user_profile.get('expertise_level', 'unknown')}
        - Interests: {', '.join(user_profile.get('interests', []))}
        - Location: {user_profile.get('location', 'Unknown')}

        Conversation History:
        {conversation_str}

        Relevant Knowledge:
        {' '.join(knowledge)}

        Current Query: {query}

        Please provide a personalized response that takes into account the user's profile,
        conversation history, and relevant knowledge. Adjust your explanation based on their
        expertise level and interests.

        Response:
        """

        # Generate the response using the MCP LLM
        response = mcp_llm(prompt)

        # Update the state with the generated response
        new_messages = state["messages"] + [
            HumanMessage(content=query),
            AIMessage(content=response)
        ]

        return {
            **state,
            "response": response,
            "messages": new_messages,
            "current_step": "done"
        }

    # Create the graph
    workflow = StateGraph(AdvancedGraphState)

    # Add nodes to the graph
    workflow.add_node("retrieve_user_profile", retrieve_user_profile)
    workflow.add_node("retrieve_conversation_history", retrieve_conversation_history)
    workflow.add_node("retrieve_knowledge", retrieve_knowledge)
    workflow.add_node("generate_response", generate_response)

    # Define the edges in the graph
    workflow.set_entry_point("retrieve_user_profile")
    workflow.add_edge("retrieve_user_profile", "retrieve_conversation_history")
    workflow.add_edge("retrieve_conversation_history", "retrieve_knowledge")
    workflow.add_edge("retrieve_knowledge", "generate_response")
    workflow.add_edge("generate_response", END)

    # Compile the graph
    graph = workflow.compile()

    # Create a memory saver for checkpointing
    memory_saver = MemorySaver()

    # Run the graph with some example queries
    users_and_queries = [
        ("user1", "Tell me about machine learning."),
        ("user1", "How does it relate to artificial intelligence?"),
        ("user2", "What is data science?"),
        ("user3", "Explain neural networks in detail.")
    ]

    # Initialize conversation histories
    conversation_histories = {
        "user1": [],
        "user2": [],
        "user3": []
    }

    for user_id, query in users_and_queries:
        print(f"\nUser: {user_id}")
        print(f"Query: {query}")

        # Get the current conversation history
        messages = conversation_histories[user_id]

        # Initialize the state
        initial_state = {
            "user_id": user_id,
            "query": query,
            "contexts": {},
            "current_step": "start",
            "response": None,
            "messages": messages
        }

        # Run the graph
        result = graph.invoke(
            initial_state,
            config={"configurable": {"thread_id": user_id}},
            checkpoint_saver=memory_saver
        )

        # Update the conversation history
        conversation_histories[user_id] = result["messages"]

        print(f"Response: {result['response']}")

    print("\n" + "=" * 50)


# Part 4: Building a Customer Support Agent with LangGraph and MCP
class Intent(str, Enum):
    """Enum for different customer support intents."""
    GREETING = "greeting"
    PRODUCT_INQUIRY = "product_inquiry"
    TECHNICAL_SUPPORT = "technical_support"
    BILLING_INQUIRY = "billing_inquiry"
    COMPLAINT = "complaint"
    UNKNOWN = "unknown"


class CustomerSupportState(TypedDict):
    """Type for the state in our customer support agent."""
    customer_id: str
    query: str
    intent: Intent
    customer_profile: Dict[str, Any]
    conversation_history: List[Dict[str, str]]
    knowledge_base_results: List[str]
    response: Optional[str]


def customer_support_agent():
    """
    Demonstrates a customer support agent built with LangGraph and MCP.

    This example shows how to build a practical customer support agent that can handle
    different types of customer inquiries.
    """
    print("\n" + "=" * 50)
    print("Customer Support Agent with LangGraph and MCP")
    print("=" * 50)

    # Initialize the MCP client
    client = MCPClient(server_url="http://localhost:8000", api_key="your_api_key_here")

    # Create an MCP LLM
    mcp_llm = MCPLLM(
        client=client,
        model_name="customer-support-model",
        model_parameters={"temperature": 0.2, "max_tokens": 400}
    )

    # Define the nodes in our graph

    # 1. Intent classification node
    def classify_intent(state: CustomerSupportState) -> CustomerSupportState:
        """Classify the intent of the customer query."""
        query = state["query"]
        print(f"Classifying intent for query: {query}")

        # Create a prompt for intent classification
        prompt = f"""
        Classify the following customer query into one of these categories:
        - greeting: General greetings or introductions
        - product_inquiry: Questions about products, features, or availability
        - technical_support: Technical issues or how-to questions
        - billing_inquiry: Questions about billing, payments, or subscriptions
        - complaint: Expressions of dissatisfaction or reporting problems
        - unknown: Queries that don't fit into any of the above categories

        Customer query: {query}

        Intent (just return the category name):
        """

        # Generate the intent classification using the MCP LLM
        intent_result = mcp_llm(prompt).strip().lower()

        # Map the result to our Intent enum
        try:
            intent = Intent(intent_result)
        except ValueError:
            intent = Intent.UNKNOWN

        print(f"Classified intent: {intent}")

        # Update the state with the classified intent
        return {**state, "intent": intent}

    # 2. Customer profile retrieval node
    def retrieve_customer_profile(state: CustomerSupportState) -> CustomerSupportState:
        """Retrieve the customer profile based on the customer ID."""
        customer_id = state["customer_id"]
        print(f"Retrieving customer profile for customer: {customer_id}")

        # In a real application, this would query a customer database
        # For this example, we'll use some hardcoded customer profiles
        customer_profiles = {
            "customer1": {
                "name": "John Smith",
                "email": "john.smith@example.com",
                "subscription_tier": "premium",
                "subscription_start_date": "2022-01-15",
                "last_purchase": "Product X Pro",
                "support_history": "3 previous technical support tickets"
            },
            "customer2": {
                "name": "Jane Doe",
                "email": "jane.doe@example.com",
                "subscription_tier": "basic",
                "subscription_start_date": "2023-03-10",
                "last_purchase": "Product Y Basic",
                "support_history": "No previous support tickets"
            },
            "customer3": {
                "name": "Robert Johnson",
                "email": "robert.johnson@example.com",
                "subscription_tier": "enterprise",
                "subscription_start_date": "2021-11-05",
                "last_purchase": "Product Z Enterprise Suite",
                "support_history": "7 previous support tickets, mostly billing related"
            }
        }

        # Get the customer profile or a default if not found
        customer_profile = customer_profiles.get(customer_id, {
            "name": "Unknown Customer",
            "subscription_tier": "unknown",
            "support_history": "No history available"
        })

        # Update the state with the customer profile
        return {**state, "customer_profile": customer_profile}

    # 3. Knowledge base query node
    def query_knowledge_base(state: CustomerSupportState) -> CustomerSupportState:
        """Query the knowledge base based on the intent and query."""
        query = state["query"]
        intent = state["intent"]
        print(f"Querying knowledge base for intent: {intent}, query: {query}")

        # In a real application, this would query a knowledge base or vector database
        # For this example, we'll use some hardcoded knowledge
        knowledge_base = {
            Intent.PRODUCT_INQUIRY: [
                "Product X is our flagship product with features A, B, and C.",
                "Product Y is our entry-level product with features A and B.",
                "Product Z is our enterprise solution with features A, B, C, and D.",
                "All products come with a 30-day money-back guarantee.",
                "Product X Pro is currently on sale with a 20% discount."
            ],
            Intent.TECHNICAL_SUPPORT: [
                "Common issue 1: Try restarting the application.",
                "Common issue 2: Check your internet connection.",
                "Common issue 3: Update to the latest version.",
                "For Product X, you can access advanced settings by going to Settings > Advanced.",
                "Our products require at least 4GB of RAM and 10GB of free disk space."
            ],
            Intent.BILLING_INQUIRY: [
                "We offer monthly and annual subscription plans.",
                "You can update your payment method in Account > Billing.",
                "Invoices are sent to your registered email address.",
                "Premium tier costs $29.99/month or $299.99/year.",
                "Enterprise tier pricing is custom and requires contacting our sales team."
            ],
            Intent.COMPLAINT: [
                "We take all customer complaints seriously.",
                "Our service level agreement guarantees 99.9% uptime.",
                "For urgent issues, you can contact our priority support line.",
                "We offer full refunds within 30 days of purchase.",
                "Our customer satisfaction team will follow up within 24 hours."
            ]
        }

        # Get relevant knowledge based on intent
        knowledge_base_results = knowledge_base.get(intent, [
            "No specific information available for this query."
        ])

        # Update the state with the knowledge base results
        return {**state, "knowledge_base_results": knowledge_base_results}

    # 4. Response generation node
    def generate_response(state: CustomerSupportState) -> CustomerSupportState:
        """Generate a response based on all available information."""
        query = state["query"]
        intent = state["intent"]
        customer_profile = state["customer_profile"]
        conversation_history = state["conversation_history"]
        knowledge_base_results = state["knowledge_base_results"]

        print(f"Generating response for intent: {intent}")

        # Format conversation history
        conversation_str = ""
        for message in conversation_history:
            conversation_str += f"Customer: {message.get('customer', '')}\n"
            conversation_str += f"Agent: {message.get('agent', '')}\n"

        # Create a prompt for the model
        prompt = f"""
        You are a customer support agent for a software company. Provide a helpful, 
        professional response to the customer query based on the following information.

        Customer Profile:
        - Name: {customer_profile.get('name', 'Unknown')}
        - Email: {customer_profile.get('email', 'Unknown')}
        - Subscription Tier: {customer_profile.get('subscription_tier', 'unknown')}
        - Subscription Start Date: {customer_profile.get('subscription_start_date', 'Unknown')}
        - Last Purchase: {customer_profile.get('last_purchase', 'Unknown')}
        - Support History: {customer_profile.get('support_history', 'No history available')}

        Conversation History:
        {conversation_str}

        Customer Query: {query}

        Query Intent: {intent}

        Relevant Knowledge Base Information:
        {' '.join(knowledge_base_results)}

        Please provide a personalized, helpful response that addresses the customer's query.
        If the intent is a greeting, be welcoming and ask how you can help.
        If the intent is a product inquiry, provide detailed product information.
        If the intent is technical support, provide troubleshooting steps.
        If the intent is a billing inquiry, provide billing information and options.
        If the intent is a complaint, be empathetic and offer solutions.

        Response:
        """

        # Generate the response using the MCP LLM
        response = mcp_llm(prompt)

        # Update the state with the generated response
        return {**state, "response": response}

    # Define the routing logic based on intent
    def router(state: CustomerSupportState) -> str:
        """Route to the appropriate node based on the intent."""
        intent = state["intent"]

        # For greeting intents, we can skip the knowledge base query
        if intent == Intent.GREETING:
            return "generate_response"
        else:
            return "query_knowledge_base"

    # Create the graph
    workflow = StateGraph(CustomerSupportState)

    # Add nodes to the graph
    workflow.add_node("classify_intent", classify_intent)
    workflow.add_node("retrieve_customer_profile", retrieve_customer_profile)
    workflow.add_node("query_knowledge_base", query_knowledge_base)
    workflow.add_node("generate_response", generate_response)

    # Define the edges in the graph
    workflow.set_entry_point("retrieve_customer_profile")
    workflow.add_edge("retrieve_customer_profile", "classify_intent")

    # Add a conditional edge from classify_intent
    workflow.add_conditional_edges(
        "classify_intent",
        router,
        {
            "query_knowledge_base": "query_knowledge_base",
            "generate_response": "generate_response"
        }
    )

    workflow.add_edge("query_knowledge_base", "generate_response")
    workflow.add_edge("generate_response", END)

    # Compile the graph
    graph = workflow.compile()

    # Create a memory saver for checkpointing
    memory_saver = MemorySaver()

    # Example customer queries
    customer_queries = [
        ("customer1", "Hi there, I'm having trouble with Product X."),
        ("customer1", "It keeps crashing when I try to open large files."),
        ("customer2", "What's the difference between Product Y and Product Z?"),
        ("customer3", "I need to update my billing information. How do I do that?"),
        ("customer3", "I'm not happy with the service. The application was down yesterday.")
    ]

    # Initialize conversation histories
    conversation_histories = {
        "customer1": [],
        "customer2": [],
        "customer3": []
    }

    for customer_id, query in customer_queries:
        print(f"\nCustomer: {customer_id}")
        print(f"Query: {query}")

        # Get the current conversation history
        conversation_history = conversation_histories[customer_id]

        # Initialize the state
        initial_state = {
            "customer_id": customer_id,
            "query": query,
            "intent": Intent.UNKNOWN,
            "customer_profile": {},
            "conversation_history": conversation_history,
            "knowledge_base_results": [],
            "response": None
        }

        # Run the graph
        result = graph.invoke(
            initial_state,
            config={"configurable": {"thread_id": customer_id}},
            checkpoint_saver=memory_saver
        )

        # Update the conversation history
        conversation_histories[customer_id].append({
            "customer": query,
            "agent": result["response"]
        })

        print(f"Response: {result['response']}")

    print("\n" + "=" * 50)


# Main function to run all examples
def main():
    """
    Main function to run all examples.
    """
    print("\n" + "=" * 50)
    print("LangGraph and MCP: Advanced Context Management")
    print("=" * 50)

    print("\nThis script demonstrates how to use LangGraph for advanced context management with MCP.")
    print("We'll explore several examples, from basic graphs to complex context-aware applications.")

    # Run the examples
    introduction_to_langgraph()
    basic_langgraph_example()
    advanced_context_management()
    customer_support_agent()

    print("\n" + "=" * 50)
    print("All examples completed!")
    print("=" * 50)


if __name__ == "__main__":
    main()
