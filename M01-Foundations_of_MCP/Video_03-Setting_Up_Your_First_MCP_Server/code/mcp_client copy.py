# mcp_client.py
# This script acts as a simple client to test the MCP server.

# --- Imports ---
# Use the same hypothetical library
try:
    import hypothetical_mcp_library as mcp
except ImportError:
    print("Error: Please install the required MCP library.")
    # Provide instructions for installing the specific MCP library here
    # e.g., print("Run: pip install actual_mcp_library_name")
    exit()

import time
import logging

# --- Configuration ---
SERVER_HOST = '127.0.0.1' # Must match the server's host
SERVER_PORT = 8765        # Must match the server's port
LOG_LEVEL = logging.INFO

# --- Logging Setup ---
logging.basicConfig(level=LOG_LEVEL, format='%(asctime)s - %(levelname)s - Client - %(message)s')
logger = logging.getLogger(__name__)

# --- Main Client Function ---
def run_client():
    """Initializes the client and sends test requests."""
    logger.info("Initializing MCP Client...")

    try:
        # --- Client Initialization ---
        # Connect to the running server
        client = mcp.Client(host=SERVER_HOST, port=SERVER_PORT)
        logger.info(f"Attempting to connect to MCP Server at {SERVER_HOST}:{SERVER_PORT}")
        # Some libraries might require an explicit connect() call
        # client.connect()
        logger.info("Connection successful (assuming init handles connection).")

    except Exception as e:
        logger.error(f"Failed to initialize or connect client: {e}", exc_info=True)
        return # Exit if connection fails

    # --- Send Test Requests ---

    # Test 1: Send a 'ping' request
    try:
        logger.info("Sending 'ping' request...")
        ping_request = {
            "command": "ping",
            "data": "Hello Server!"
        }
        # The send_request method signature depends on the library
        response = client.send_request(ping_request) # Or client.call("ping", data=...)
        logger.info(f"Received response for 'ping': {response}")

    except Exception as e:
        logger.error(f"Error during 'ping' request: {e}", exc_info=True)

    # Wait a moment
    time.sleep(1)

    # Test 2: Send a 'get_context' request
    try:
        logger.info("Sending 'get_context' request for session '12345'...")
        context_request = {
            "command": "get_context",
            "session_id": "12345"
        }
        response = client.send_request(context_request)
        logger.info(f"Received response for 'get_context': {response}")

    except Exception as e:
        logger.error(f"Error during 'get_context' request: {e}", exc_info=True)

    # --- Close Connection (if necessary) ---
    # Some libraries might require explicit disconnection
    try:
        # client.close()
        logger.info("Client connection closed (if applicable).")
    except AttributeError:
        logger.info("Client does not require explicit close or method not found.")
    except Exception as e:
         logger.error(f"Error closing client connection: {e}", exc_info=True)


# --- Script Execution ---
if __name__ == "__main__":
    run_client()

