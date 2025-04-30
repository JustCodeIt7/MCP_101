import subprocess
import sys
from pathlib import Path
import json
import argparse
import requests
from mcp import ClientSession, stdio_client

# Ollama API endpoint
OLLAMA_API_URL = "http://localhost:11434/api/chat"

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Ollama client with MCP context")
    parser.add_argument("--user", default="alice", help="User ID for context (default: alice)")
    parser.add_argument("--question", default="Explain GPUs", help="Question to ask (default: Explain GPUs)")
    parser.add_argument("--model", default="llama2", help="Ollama model to use (default: llama2)")
    args = parser.parse_args()
    
    # Get the path to the context server from the previous example
    context_server_dir = Path(__file__).parent.parent / "10-Chatbot_Context_Server_Client"
    server_path = context_server_dir / "server.py"
    
    print(f"Looking for context server at: {server_path}")
    
    if not server_path.exists():
        print(f"Error: Context server not found at {server_path}")
        print("Please ensure the Chatbot Context Server is set up correctly.")
        return
    
    # Start the context server as a subprocess
    print(f"Starting context server...")
    server_process = subprocess.Popen(
        [sys.executable, str(server_path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )
    
    try:
        # Connect to the server
        with ClientSession(stdio_client.spawn(f"python {server_path}")) as session:
            # Get the user preferences
            user_id = args.user
            question = args.question
            model = args.model
            
            print(f"Getting preferences for user: {user_id}")
            prefs = session.read_resource(f"prefs://{user_id}")
            
            print(f"User preferences: {json.dumps(prefs, indent=2)}")
            
            # Build the system message based on user preferences
            system_message = build_system_message(prefs)
            
            print("\nGenerated system message:")
            print("-" * 40)
            print(system_message)
            print("-" * 40)
            
            # Call the Ollama API
            print(f"\nCalling Ollama API with model: {model}...")
            
            try:
                # Check if Ollama is running
                response = call_ollama(model, system_message, question)
                print("\nResponse:")
                print("=" * 60)
                print(response)
                print("=" * 60)
            except Exception as e:
                print(f"\nError calling Ollama API: {str(e)}")
                print("\nIs Ollama running? You can start it with 'ollama serve' and ensure the model is available with 'ollama pull llama2'")
                print("\nSimulating response instead...")
                response = simulate_response(system_message, question, prefs)
                print("\nSimulated Response:")
                print("=" * 60)
                print(response)
                print("=" * 60)
            
    finally:
        # Clean up the server process
        print("Shutting down context server...")
        server_process.terminate()

def build_system_message(prefs):
    """
    Build a system message based on user preferences.
    
    Args:
        prefs: User preferences dictionary
        
    Returns:
        A formatted system message string
    """
    # Extract preferences
    topic = prefs.get("topic", "general")
    tone = prefs.get("tone", "neutral")
    detail_level = prefs.get("detail_level", "medium")
    include_examples = prefs.get("examples", False)
    
    # Build the system message
    system_message = f"You are an AI assistant that specializes in {topic}. "
    system_message += f"Please respond in a {tone} tone. "
    
    if detail_level == "low":
        system_message += "Keep your explanations brief and to the point. "
    elif detail_level == "high":
        system_message += "Provide detailed and comprehensive explanations. "
    else:  # medium
        system_message += "Provide moderately detailed explanations. "
    
    if include_examples:
        system_message += "Include practical examples in your response. "
    
    return system_message

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
        "stream": False  # Set to True to stream the response
    }
    
    # Make the API request
    response = requests.post(OLLAMA_API_URL, json=payload)
    
    # Check if the request was successful
    if response.status_code == 200:
        return response.json()["message"]["content"]
    else:
        raise Exception(f"API request failed with status code {response.status_code}: {response.text}")

def simulate_response(system_message, question, prefs):
    """
    Simulate an Ollama response for demonstration purposes.
    
    Args:
        system_message: The system message
        question: The user's question
        prefs: User preferences
        
    Returns:
        A simulated response
    """
    tone = prefs.get("tone", "neutral")
    
    if "gpu" in question.lower() or "gpus" in question.lower():
        if tone == "technical":
            return """
GPUs (Graphics Processing Units) are specialized processors designed for parallel computation, particularly for rendering graphics. They differ from CPUs in their architecture, which consists of thousands of smaller cores optimized for handling multiple tasks simultaneously.

Key components include:
- CUDA cores/Stream processors: The basic computational units
- Texture mapping units: Handle texture operations
- Render output units: Process final pixel output
- Video memory (VRAM): Dedicated high-speed memory

GPUs excel at matrix operations and floating-point calculations, making them suitable for:
1. Graphics rendering
2. Machine learning
3. Scientific computing
4. Cryptocurrency mining

Their parallel architecture allows them to process large data sets simultaneously, which is why they've become essential for deep learning applications.
"""
        elif tone == "enthusiastic":
            return """
GPUs are AMAZING pieces of technology! They're like the unsung heroes of modern computing!

Unlike CPUs which handle one task at a time really well, GPUs can handle THOUSANDS of smaller tasks all at once! This parallel processing ability makes them PERFECT for graphics (which is what they were designed for) but also for so many other exciting applications!

The way they're structured is fascinating - instead of a few powerful cores, they have THOUSANDS of smaller cores that work together in harmony! This design is what gives them their incredible power for certain types of calculations!

What's really exciting is how they've revolutionized fields like AI and machine learning! Tasks that would take days on a CPU can be done in HOURS or even MINUTES on a good GPU!

Every time you enjoy amazing graphics in a video game or use an AI application, you're experiencing the INCREDIBLE power of GPUs at work!
"""
        else:
            return """
GPUs (Graphics Processing Units) are specialized electronic circuits designed primarily to process and render graphics quickly. Unlike CPUs (Central Processing Units) which are designed for general-purpose computing, GPUs are optimized for parallel processing of many operations simultaneously.

The main characteristics of GPUs include:

1. Parallel architecture: GPUs contain many small processing cores that can handle multiple tasks at the same time.

2. Specialized for graphics: They excel at the mathematical calculations needed for rendering images, textures, and 3D models.

3. High memory bandwidth: GPUs can access their dedicated memory much faster than CPUs can access RAM.

Originally developed for gaming and graphics applications, GPUs have found uses in many other fields including artificial intelligence, scientific simulations, and cryptocurrency mining due to their ability to perform many calculations in parallel.

Modern computers typically contain both a CPU and a GPU, with the CPU handling general computing tasks and the GPU handling graphics rendering and other specialized parallel processing tasks.
"""
    else:
        return f"This is a simulated response to your question about {question}. In a real implementation, this would be generated by the Ollama API using the {prefs.get('tone', 'neutral')} tone you requested."

if __name__ == "__main__":
    main()