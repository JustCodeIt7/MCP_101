import subprocess
import json
import time
import threading
import sys
import os

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
server_script_path = os.path.join(script_dir, "server.py")

# --- Communication Functions ---
def send_request(process, request_data):
    """Sends a JSON request to the MCP server process."""
    request_str = json.dumps(request_data) + "\n"
    try:
        process.stdin.write(request_str.encode('utf-8'))
        process.stdin.flush()
        print(f"CLIENT -> SERVER: {request_str.strip()}")
    except BrokenPipeError:
        print("Error: Could not write to server stdin. It might have terminated.")
        sys.exit(1)
    except Exception as e:
        print(f"Error sending request: {e}")
        sys.exit(1)

def read_output(process, output_queue):
    """Reads JSON responses from the MCP server process's stdout."""
    try:
        for line in iter(process.stdout.readline, b''):
            line_str = line.decode('utf-8').strip()
            if line_str:
                print(f"SERVER -> CLIENT: {line_str}")
                try:
                    response_data = json.loads(line_str)
                    output_queue.put(response_data)
                except json.JSONDecodeError:
                    print(f"Warning: Received non-JSON line from server: {line_str}")
            else:
                # Handle empty lines if necessary, or just ignore
                pass
        print("Server stdout stream ended.")
    except Exception as e:
        print(f"Error reading server output: {e}")
    finally:
        # Signal that reading is done (optional, depends on main loop logic)
        output_queue.put(None) # Use None as a sentinel value

# --- Main Execution ---
if __name__ == "__main__":
    print(f"Starting MCP server: {server_script_path}")
    # Start the server process
    try:
        server_process = subprocess.Popen(
            [sys.executable, server_script_path], # Use sys.executable to ensure same python interpreter
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, # Capture stderr as well
            text=False, # Use binary mode for streams
            bufsize=1, # Line buffered
            cwd=script_dir # Run server in its own directory
        )
    except FileNotFoundError:
        print(f"Error: Could not find server script at {server_script_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error starting server process: {e}")
        sys.exit(1)

    print(f"Server process started (PID: {server_process.pid}). Waiting for it to initialize...")
    time.sleep(2) # Give the server a moment to start up

    # Check if server started correctly
    if server_process.poll() is not None:
        print("Server process terminated unexpectedly.")
        stderr_output = server_process.stderr.read().decode('utf-8')
        print("--- Server Stderr ---")
        print(stderr_output)
        print("---------------------")
        sys.exit(1)


    # Use a queue to safely pass data from the reader thread
    from queue import Queue
    response_queue = Queue()

    # Start a thread to read server output non-blockingly
    reader_thread = threading.Thread(target=read_output, args=(server_process, response_queue), daemon=True)
    reader_thread.start()

    request_id_counter = 1

    # --- Test Cases ---

    # 1. Call 'add' tool
    print("\n--- Test 1: Calling 'add' tool ---")
    add_request = {
        "mcp_protocol_version": "1.0",
        "request_id": str(request_id_counter),
        "type": "tool_call",
        "tool_name": "add",
        "arguments": {"a": 5, "b": 7}
    }
    request_id_counter += 1
    send_request(server_process, add_request)
    time.sleep(0.5) # Wait for response

    # 2. Access 'users://all' resource
    print("\n--- Test 2: Accessing 'users://all' resource ---")
    get_all_request = {
        "mcp_protocol_version": "1.0",
        "request_id": str(request_id_counter),
        "type": "resource_access",
        "uri": "users://all"
    }
    request_id_counter += 1
    send_request(server_process, get_all_request)
    time.sleep(0.5)

    # 3. Access specific user 'users://alice/profile'
    print("\n--- Test 3: Accessing 'users://alice/profile' resource ---")
    get_alice_request = {
        "mcp_protocol_version": "1.0",
        "request_id": str(request_id_counter),
        "type": "resource_access",
        "uri": "users://alice/profile"
    }
    request_id_counter += 1
    send_request(server_process, get_alice_request)
    time.sleep(0.5)

    # 4. Add a new user 'david'
    print("\n--- Test 4: Calling 'add_user' tool ---")
    add_david_request = {
        "mcp_protocol_version": "1.0",
        "request_id": str(request_id_counter),
        "type": "tool_call",
        "tool_name": "add_user",
        "arguments": {
            "user_id": "david",
            "name": "David Lee",
            "email": "david@example.com",
            "age": 40,
            "interests": ["DevOps", "Kubernetes"]
        }
    }
    request_id_counter += 1
    send_request(server_process, add_david_request)
    time.sleep(0.5)

    # 5. Access 'users://all' again to see David
    print("\n--- Test 5: Accessing 'users://all' resource again ---")
    get_all_request_2 = {
        "mcp_protocol_version": "1.0",
        "request_id": str(request_id_counter),
        "type": "resource_access",
        "uri": "users://all"
    }
    request_id_counter += 1
    send_request(server_process, get_all_request_2)
    time.sleep(0.5)

    # 6. Remove user 'bob'
    print("\n--- Test 6: Calling 'remove_user' tool ---")
    remove_bob_request = {
        "mcp_protocol_version": "1.0",
        "request_id": str(request_id_counter),
        "type": "tool_call",
        "tool_name": "remove_user",
        "arguments": {"user_id": "bob"}
    }
    request_id_counter += 1
    send_request(server_process, remove_bob_request)
    time.sleep(0.5)

    # 7. Access 'users://all' one last time
    print("\n--- Test 7: Accessing 'users://all' resource finally ---")
    get_all_request_3 = {
        "mcp_protocol_version": "1.0",
        "request_id": str(request_id_counter),
        "type": "resource_access",
        "uri": "users://all"
    }
    request_id_counter += 1
    send_request(server_process, get_all_request_3)
    time.sleep(1) # Longer wait to ensure final response is captured

    # --- Cleanup ---
    print("\n--- Tests complete. Terminating server process. ---")
    try:
        # Close stdin first to signal no more input
        server_process.stdin.close()
    except Exception as e:
        print(f"Error closing server stdin: {e}")

    # Wait for the reader thread to finish processing any remaining output
    reader_thread.join(timeout=2) # Wait up to 2 seconds for the reader

    # Terminate the server process if it's still running
    if server_process.poll() is None:
        print("Force terminating server process...")
        server_process.terminate()
        try:
            server_process.wait(timeout=5) # Wait for termination
        except subprocess.TimeoutExpired:
            print("Server did not terminate gracefully, killing.")
            server_process.kill()

    print("Server process finished.")

    # Print any remaining stderr output
    stderr_output = server_process.stderr.read().decode('utf-8', errors='ignore')
    if stderr_output:
        print("--- Server Stderr Output ---")
        print(stderr_output)
        print("--------------------------")

    # Drain and print any remaining items from the queue (optional, for debugging)
    # print("\n--- Remaining items in queue: ---")
    # while not response_queue.empty():
    #     item = response_queue.get_nowait()
    #     print(item)
    # print("-------------------------------")

    print("Client finished.")
