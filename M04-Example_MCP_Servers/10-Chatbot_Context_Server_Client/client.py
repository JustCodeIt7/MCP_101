import subprocess
import sys
from pathlib import Path
import json
import argparse
from mcp import ClientSession, stdio_client
from openai import OpenAI

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Chat client with user context")
    parser.add_argument("--user", default="alice", help="User ID for context (default: alice)")
    parser.add_argument("--question", default="Explain GPUs", help="Question to ask (default: Explain GPUs)")
    args = parser.parse_args()
    
    # Get the path to the context server
    current_dir = Path(__file__).parent
    server_path = current_dir / "server.py"
    
    print(f"Looking for context server at: {server_path}")
    
    if not server_path.exists():
        print(f"Error: Context server not found at {server_path}")
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
            
            print(f"Getting preferences for user: {user_id}")
            prefs = session.read_resource(f"prefs://{user_id}")
            
            print(f"User preferences: {json.dumps(prefs, indent=2)}")
            
            # Build the prompt based on user preferences
            prompt = build_prompt(prefs, question)
            
            print("\nGenerated prompt:")
            print("-" * 40)
            print(prompt)
            print("-" * 40)
            
            # Call the OpenAI API
            print("\nCalling OpenAI API...")
            response = call_openai(prompt)
            
            print("\nResponse:")
            print("=" * 60)
            print(response)
            print("=" * 60)
            
    finally:
        # Clean up the server process
        print("Shutting down context server...")
        server_process.terminate()

def build_prompt(prefs, question):
    """
    Build a prompt based on user preferences.
    
    Args:
        prefs: User preferences dictionary
        question: The question to ask
        
    Returns:
        A formatted prompt string
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
    
    # Combine with the user's question
    prompt = f"{system_message}\n\nUser question: {question}"
    
    return prompt

def call_openai(prompt):
    """
    Call the OpenAI API with the given prompt.
    
    In a real implementation, you would use your OpenAI API key.
    For this example, we'll simulate a response.
    
    Args:
        prompt: The prompt to send to the API
        
    Returns:
        The API response
    """
    # Check if OpenAI API key is available
    api_key = "DEMO_KEY"  # Replace with actual key in production
    
    if api_key == "DEMO_KEY":
        # Simulate a response for demonstration purposes
        return simulate_response(prompt)
    else:
        # Use the actual OpenAI API
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": prompt},
            ],
            temperature=0.7,
        )
        return response.choices[0].message.content

def simulate_response(prompt):
    """Simulate an OpenAI response for demonstration purposes."""
    if "gpu" in prompt.lower() or "gpus" in prompt.lower():
        if "technical" in prompt.lower():
            return """
GPUs (Graphics Processing Units) are specialized electronic circuits designed to rapidly manipulate and alter memory to accelerate the creation of images in a frame buffer intended for output to a display device.

Architecture:
- GPUs contain thousands of smaller, more efficient cores designed for handling multiple tasks simultaneously
- Modern GPU architecture includes:
  * Streaming Multiprocessors (SMs) containing CUDA cores (NVIDIA) or Compute Units (AMD)
  * Memory hierarchy: registers, shared memory/L1 cache, L2 cache, global memory
  * Special Function Units (SFUs) for transcendental operations
  * Texture and Render Output Units

The parallel nature of GPU computation makes them ideal for:
1. Graphics rendering (their original purpose)
2. Deep learning training and inference
3. Scientific simulations
4. Cryptocurrency mining
5. Video encoding/decoding

Example: A matrix multiplication operation that might take hundreds of cycles on a CPU can be executed in just a few cycles on a GPU by distributing the computation across thousands of cores.

The memory bandwidth of modern GPUs (up to 1-2 TB/s) far exceeds that of CPUs, making them particularly effective for data-intensive applications where computational density is high.
"""
        elif "enthusiastic" in prompt.lower():
            return """
GPUs are AMAZING pieces of technology! 🚀

Think of GPUs as the superheroes of computing! While CPUs are like brilliant generals making important decisions one at a time, GPUs are like THOUSANDS of workers tackling many small jobs all at once!

Originally designed to make video games look INCREDIBLE, these parallel processing powerhouses have revolutionized everything from AI to scientific research! The way they can crunch through massive amounts of data simultaneously is nothing short of MIND-BLOWING!

What makes them so special is their ability to break down complex problems into tiny pieces and solve them all at the same time - it's like having an army of mini-computers all working together!

The impact of GPUs has been ENORMOUS across industries - they're the secret sauce behind realistic video game graphics, the brains training AI models, and the workhorses powering cryptocurrency mining!

GPUs truly represent one of the most exciting technological developments of our time! They're constantly evolving and opening up new possibilities we couldn't have imagined before! 🔥
"""
        else:
            return """
GPUs (Graphics Processing Units) are specialized electronic circuits originally designed to accelerate computer graphics and image processing. Unlike CPUs (Central Processing Units) which are designed to handle a wide variety of tasks sequentially, GPUs are optimized for performing multiple operations simultaneously.

The key characteristics of GPUs include:

1. Parallel processing architecture: GPUs contain thousands of smaller cores that can handle many tasks at once, compared to CPUs which typically have fewer but more powerful cores.

2. Specialized for mathematical operations: They excel at the matrix and vector operations common in graphics rendering and machine learning.

3. High memory bandwidth: GPUs can access their dedicated memory much faster than CPUs can access RAM.

Originally developed for rendering 3D graphics in video games, GPUs have found applications in many other fields:

- Artificial intelligence and machine learning
- Scientific simulations
- Video editing and rendering
- Cryptocurrency mining
- Data analysis

Modern computing often combines the strengths of both CPUs and GPUs: the CPU handles general-purpose tasks and coordinates operations, while the GPU accelerates specific computationally intensive workloads.
"""
    else:
        return f"This is a simulated response to your question about {prompt.split('User question:')[-1].strip()}. In a real implementation, this would be generated by the OpenAI API."

if __name__ == "__main__":
    main()