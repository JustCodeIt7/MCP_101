#!/usr/bin/env python3
"""
MCP Server Test with Ollama

This script demonstrates how to test an MCP server (specifically the User Profile MCP server)
using Ollama as the LLM. It allows you to:
1. Retrieve user profiles from the server
2. Add new users to the server
3. Remove users from the server
4. Ask Ollama to process the profile information

Prerequisites:
- Python 3.7+
- mcp library installed (pip install mcp)
- requests library installed (pip install requests)
- Ollama installed and running locally (https://ollama.com/)
- A model pulled in Ollama (e.g., llama2)
"""

import subprocess
import sys
import json
import argparse
from pathlib import Path
import requests
import time
from mcp import ClientSession, stdio_client


# Ollama API endpoint - update if your Ollama instance runs on a different URL
OLLAMA_API_URL = "http://localhost:11434/api/chat"

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Test MCP User Profile server with Ollama")
    parser.add_argument("--model", default="llama2", help="Ollama model to use (default: llama2)")
    parser.add_argument("--user", default="alice", help="User ID to query (default: alice)")
    parser.add_argument("--action", default="query",
                        choices=["query", "add", "remove", "list", "calculate"],
                        help="Action to perform (default: query)")
    parser.add_argument("--name", help="User name (for add action)")
    parser.add_argument("--email", help="User email (for add action)")
    parser.add_argument("--age", type=int, help="User age (for add action)")
    parser.add_argument("--interests", help="User interests, comma-separated (for add action)")
    parser.add_argument("--a", type=int, default=5, help="First number for calculate action (default: 5)")
    parser.add_argument("--b", type=int, default=3, help="Second number for calculate action (default: 3)")
    parser.add_argument("--question", default="What can you tell me about this user?", 
                      help="Question to ask about the user profile (default: What can you tell me about this user?)")
    args = parser.parse_args()
    
    # Path to the server script
    server_path = Path("M04-Example_MCP_Servers/02-User_Profile_MCP_Server/server.py")
    
    print(f"Looking for User Profile MCP server at: {server_path}")
    
    if not server_path.exists():
        print(f"Error: User Profile MCP server not found at {server_path}")
        print("Please ensure the User Profile MCP Server is in the correct location.")
        return
     
    # Start the server as a subprocess
    print(f"Starting User Profile MCP server...")
    server_process = subprocess.Popen(
        [sys.executable, str(server_path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )
    
    try:
        # Give the server a moment to start up
        time.sleep(1)
        
        # Connect to the server using MCP ClientSession
        with ClientSession(stdio_client.spawn(f"python {server_path}")) as session:
            print("\nConnected to User Profile MCP server")
            
            # List available tools and resources
            tools = session.list_tools()
            print("\nAvailable tools:")
            print(json.dumps(tools, indent=2))
            
            if args.action == "query":
                # Get user profile
                user_id = args.user
                print(f"\nRetrieving profile for user: {user_id}")
                
                try:
                    profile = session.read_resource(f"users://{user_id}/profile")
                    
                    if "error" in profile:
                        print(f"Error: {profile['error']}")
                        return
                    
                    print("\nUser Profile:")
                    print(json.dumps(profile, indent=2))
                    
                    # Process the profile with Ollama
                    process_with_ollama(args.model, profile, args.question)
                    
                except Exception as e:
                    print(f"Error retrieving user profile: {str(e)}")
            
            elif args.action == "list":
                # List all users
                print("\nRetrieving all users")
                
                try:
                    all_users = session.read_resource("users://all")
                    print("\nAll Users:")
                    print(json.dumps(all_users, indent=2))
                except Exception as e:
                    print(f"Error retrieving all users: {str(e)}")
            
            elif args.action == "add":
                # Check required parameters
                if not all([args.name, args.email, args.age, args.interests]):
                    print("Error: For 'add' action, you must provide --name, --email, --age, and --interests")
                    return
                
                # Add a new user
                print(f"\nAdding new user: {args.user}")
                
                try:
                    interests = [i.strip() for i in args.interests.split(',')]
                    result = session.call_tool("add_user", 
                                              user_id=args.user, 
                                              name=args.name, 
                                              email=args.email, 
                                              age=args.age, 
                                              interests=interests)
                    
                    print("\nResult:")
                    print(json.dumps(result, indent=2))
                except Exception as e:
                    print(f"Error adding user: {str(e)}")
            
            elif args.action == "remove":
                # Remove a user
                print(f"\nRemoving user: {args.user}")
                
                try:
                    result = session.call_tool("remove_user", user_id=args.user)
                    print("\nResult:")
                    print(json.dumps(result, indent=2))
                except Exception as e:
                    print(f"Error removing user: {str(e)}")
            
            elif args.action == "calculate":
                # Test the add function
                print(f"\nTesting add function with a={args.a}, b={args.b}")
                
                try:
                    result = session.call_tool("add", a=args.a, b=args.b)
                    print(f"\nResult: {args.a} + {args.b} = {result}")
                except Exception as e:
                    print(f"Error calling add function: {str(e)}")
    
    finally:
        # Clean up the server process
        print("\nShutting down User Profile MCP server...")
        server_process.terminate()

def process_with_ollama(model, profile, question):
    """
    Process a user profile with Ollama.
    
    Args:
        model: The Ollama model to use
        profile: The user profile as a dictionary
        question: The question to ask about the profile
    """
    print(f"\nProcessing profile with Ollama (model: {model})...")
    
    # Create a prompt from the profile
    system_message = f"""
    You are an assistant that analyzes user profiles. You have access to the following user profile:
    
    Name: {profile.get('name')}
    Email: {profile.get('email')}
    Age: {profile.get('age')}
    Interests: {', '.join(profile.get('interests', []))}
    
    Please provide helpful and insightful information based on this profile.
    """

    
    try:
        # Call Ollama API
        response = call_ollama(model, system_message, question)
        print("\nOllama Response:")
        print("=" * 60)
        print(response)
        print("=" * 60)
    except Exception as e:
        print(f"\nError calling Ollama API: {str(e)}")
        print("\nIs Ollama running? You can start it with 'ollama serve' and ensure the model is available with 'ollama pull llama2'")
        print("\nSimulating response instead...")
        
        # Provide a simulated response if Ollama is not available
        simulated_response = simulate_response(profile, question)
        print("\nSimulated Response:")
        print("=" * 60)
        print(simulated_response)
        print("=" * 60)

def call_ollama(model, system_message, question):
    """
    Call the Ollama API with the given model, system message, and question.
    
    Args:
        model: The Ollama model to use (e.g., "llama2")
        system_message: The system message providing context
        question: The user's question
        
    Returns:
        The API response
    """
    # Prepare the request payload
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": question}
        ],
        "stream": False  # Set to True if you want to stream the response
    }
    
    # Make the API request
    response = requests.post(OLLAMA_API_URL, json=payload)
    
    # Check if the request was successful
    if response.status_code == 200:
        return response.json()["message"]["content"]
    else:
        raise Exception(f"API request failed with status code {response.status_code}: {response.text}")

def simulate_response(profile, question):
    """
    Simulate an Ollama response for demonstration purposes.
    
    Args:
        profile: The user profile
        question: The question about the profile
        
    Returns:
        A simulated response
    """
    name = profile.get('name', 'Unknown')
    age = profile.get('age', 'Unknown')
    interests = profile.get('interests', [])
    
    return f"""
Based on the profile information provided, {name} is {age} years old and has interests in {', '.join(interests)}.

Given the interests in {', '.join(interests)}, this person seems to be technically inclined and engaged with modern technology and development. They might work in the tech industry or be a student in a related field.

If I were to recommend resources or topics that might interest {name}, I would suggest:

1. Online courses or workshops related to {interests[0] if interests else 'their interests'}
2. Community groups or forums focused on {interests[1] if len(interests) > 1 else 'their technical interests'}
3. Recent developments in {interests[2] if len(interests) > 2 else 'technology fields'}

This person might benefit from connecting with others who share similar interests, as their profile suggests a curious and learning-oriented mindset.
"""

if __name__ == "__main__":
    main()