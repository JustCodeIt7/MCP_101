import argparse
import ollama
from mcp.client import Client
import json
import sys
import logging
import shlex # For potentially safer parsing if not using JSON

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_mcp_schema(client: Client) -> str:
    """Generates a description of available MCP tools and resources for Ollama."""
    schema = "Available MCP Functions:\n\n"
    schema += "Tools (callable functions):\n"
    if not client.tools:
        schema += "  (No tools available)\n"
    for name, tool in client.tools.items():
        # Use repr(signature) to get a string like '(a: int, b: int)'
        sig_repr = repr(tool.signature)
        # Clean up signature string for better readability if needed
        # sig_repr = sig_repr.replace("'", "") # Example cleanup
        schema += f"- Name: {name}\n"
        schema += f"  Signature: {sig_repr}\n"
        schema += f"  Description: {tool.description}\n"

    schema += "\nResources (data endpoints, primarily GET supported here):\n"
    if not client.resources:
        schema += "  (No resources available)\n"
    for name, resource in client.resources.items():
        # Assuming resources are simple GET for now
        schema += f"- Name: {name}\n"
        schema += f"  Description: {resource.description}\n"
        schema += f"  Methods: GET\n" # Explicitly state supported method

    return schema

def parse_ollama_response_json(response: str) -> tuple[str | None, str | None, dict | None]:
    """
    Parses Ollama's response expecting a JSON object for function calls.
    Returns (call_type, name, args) or (None, None, None)
    call_type: 'tool' or 'resource'
    name: name of the tool or resource URI
    args: dictionary of arguments for tools, or None for resources (GET)
    """
    response = response.strip()
    try:
        # Handle potential markdown code blocks around JSON
        if response.startswith("```json"):
            response = response[7:]
        if response.startswith("```"):
            response = response[3:]
        if response.endswith("```"):
            response = response[:-3]
        response = response.strip()

        call_info = json.loads(response)
        if isinstance(call_info, dict) and "type" in call_info and "name" in call_info:
            call_type = call_info.get("type")
            name = call_info.get("name")

            if call_type == "tool":
                args = call_info.get("args", {})
                if isinstance(args, dict):
                    return "tool", name, args
                else:
                     logging.warning(f"Invalid 'args' format in tool call: {call_info}")
                     return None, None, None
            elif call_type == "resource":
                 # For now, assume GET method if not specified or if explicitly GET
                 method = call_info.get("method", "GET")
                 if method.upper() == "GET":
                     # Return resource name and None for args (as it's a GET)
                     return "resource", name, None
                 else:
                     logging.warning(f"Unsupported resource method '{method}' in call: {call_info}")
                     return None, None, None
            else:
                logging.warning(f"Unknown call type '{call_type}' in response: {call_info}")
                return None, None, None
        else:
            # Not a valid JSON call structure
            return None, None, None
    except json.JSONDecodeError:
        # Not JSON, assume it's a direct text response from Ollama
        return None, None, None
    except Exception as e:
        logging.warning(f"Error parsing Ollama JSON response '{response}': {e}")
        return None, None, None


def main():
    parser = argparse.ArgumentParser(description="Chat with an MCP server via Ollama.")
    parser.add_argument("mcp_server",default='http://127.0.0.1:6274', help="MCP server address (e.g., stdio, tcp://localhost:5000)")
    parser.add_argument("--ollama-host", default="http://localhost:11434", help="Ollama API host (default: http://localhost:11434)")
    parser.add_argument("--ollama-model", default="llama3.2", help="Ollama model to use (default: llama3)")
    args = parser.parse_args()

    try:
        logging.info(f"Connecting to MCP server: {args.mcp_server}")
        # Increase timeout if stdio connection takes time
        mcp_client = Client(args.mcp_server, connection_timeout=15)
        mcp_client.wait_for_connection(timeout=10) # Wait for stdio connection if needed
        logging.info("MCP client connected.")
    except Exception as e:
        logging.error(f"Failed to connect to MCP server '{args.mcp_server}': {e}")
        sys.exit(1)

    try:
        logging.info(f"Initializing Ollama client (Host: {args.ollama_host}, Model: {args.ollama_model})")
        ollama_client = ollama.Client(host=args.ollama_host)
        # Check connection and model availability
        ollama_client.list()
        logging.info("Ollama client initialized.")
    except Exception as e:
        logging.error(f"Failed to initialize Ollama client or list models: {e}")
        mcp_client.close()
        sys.exit(1)

    try:
        mcp_schema = get_mcp_schema(mcp_client)
        logging.info("MCP Schema retrieved.")
        # print(f"DEBUG: MCP Schema:\n{mcp_schema}") # Optional debug print

        system_prompt = f"""You are an assistant that can interact with an MCP server based on user requests.
You have access to the following MCP functions (tools and resources):
{mcp_schema}

To respond to the user, you have two options:
1. Provide a direct natural language answer if the user's request does not require using an MCP function.
2. If you need to call an MCP function, respond ONLY with a single JSON object describing the call. Do NOT add any explanatory text before or after the JSON.

JSON format for calling a TOOL:
{{"type": "tool", "name": "tool_name", "args": {{"arg1": value1, "arg2": "value2", ...}}}}

JSON format for calling a RESOURCE (only GET is supported):
{{"type": "resource", "name": "resource_uri", "method": "GET"}}

Example Tool Call JSON: {{"type": "tool", "name": "add", "args": {{"a": 5, "b": 10}}}}
Example Resource Call JSON: {{"type": "resource", "name": "users://alice/profile", "method": "GET"}}

Carefully examine the user's request and the available functions. Choose the appropriate function and construct the JSON precisely, or provide a textual response.
"""

        messages = [{"role": "system", "content": system_prompt}]

        print(f"\nConnected to MCP Server ({args.mcp_server}) and Ollama ({args.ollama_model} on {args.ollama_host}).")
        print("Type 'quit' or 'exit' to end the session.")
        while True:
            user_input = input("You: ")
            if user_input.lower() in ['quit', 'exit']:
                break

            messages.append({"role": "user", "content": user_input})

            try:
                logging.info("Sending request to Ollama...")
                response = ollama_client.chat(
                    model=args.ollama_model,
                    messages=messages
                )
                ollama_response_content = response['message']['content']
                logging.info(f"Ollama raw response: {ollama_response_content}")

                # Add Ollama's response to history *before* potential modification by tool result
                messages.append({"role": "assistant", "content": ollama_response_content})

                # --- Attempt to parse and execute MCP call ---
                call_type, func_name, func_args = parse_ollama_response_json(ollama_response_content)

                if call_type and func_name:
                    logging.info(f"Ollama requested MCP call: Type={call_type}, Name={func_name}, Args={func_args}")
                    mcp_result = None
                    result_str = ""
                    try:
                        if call_type == "tool":
                            if func_name in mcp_client.tools:
                                logging.info(f"Executing tool: {func_name}(**{func_args})")
                                mcp_result = mcp_client.toolsfunc_name
                                result_str = f"MCP Tool Result ({func_name}): {json.dumps(mcp_result)}"
                                logging.info(f"Tool '{func_name}' executed successfully.")
                            else:
                                result_str = f"Error: Tool '{func_name}' not found on MCP server."
                                logging.warning(result_str)
                        elif call_type == "resource": # Assumed GET
                             if func_name in mcp_client.resources:
                                 logging.info(f"Executing resource GET: {func_name}")
                                 mcp_result = mcp_client.resources[func_name].get()
                                 result_str = f"MCP Resource Result ({func_name}): {json.dumps(mcp_result)}"
                                 logging.info(f"Resource '{func_name}' GET executed successfully.")
                             else:
                                 result_str = f"Error: Resource '{func_name}' not found on MCP server."
                                 logging.warning(result_str)

                        print(result_str)
                        # Add execution result back into the conversation history for context
                        messages.append({"role": "user", "content": result_str}) # Feed result back as if user provided it

                    except Exception as e:
                        error_msg = f"Error executing MCP call '{func_name}': {e}"
                        logging.exception(error_msg) # Log full traceback
                        print(error_msg)
                        # Add error message to history
                        messages.append({"role": "user", "content": f"Error during MCP execution: {e}"})

                else:
                    # Ollama provided a direct answer (not valid JSON function call)
                    print(f"Ollama: {ollama_response_content}")

                # Optional: Limit message history size
                # MAX_HISTORY = 15
                # if len(messages) > MAX_HISTORY:
                #     messages = [messages[0]] + messages[-(MAX_HISTORY-1):]

            except Exception as e:
                logging.exception("Error during Ollama interaction:") # Log full traceback
                print(f"An error occurred during Ollama interaction: {e}")
                # Prevent broken state by potentially removing last user message if Ollama call failed
                if messages and messages[-1]["role"] == "user":
                     # Check if the last assistant message was added before deciding to pop
                     if len(messages) > 1 and messages[-2]["role"] == "assistant":
                         pass # Ollama response was added, keep user message
                     else:
                         messages.pop() # Remove user message that caused failure

    finally:
        logging.info("Closing MCP client connection.")
        mcp_client.close()
        print("\nConnection closed. Goodbye!")

if __name__ == "__main__":
    main()
