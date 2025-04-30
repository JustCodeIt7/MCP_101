"""
# MCP Clients: Accessing and Using MCP Servers

This script demonstrates how to set up and use MCP (Model-Centric Programming) clients
to interact with MCP servers. It covers client initialization, sending requests,
and handling server responses.

## Prerequisites
- Python 3.7+
- An MCP server running (see Video 3)
- Required packages: requests, json

## Key Concepts
1. Initializing an MCP client
2. Sending requests to MCP servers
3. Handling server responses
4. Error handling in client-server communication
"""

import requests
import json
import time
from typing import Dict, Any, List, Optional, Union


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

        # Test connection to server
        self.test_connection()

    def test_connection(self) -> bool:
        """
        Test the connection to the MCP server.

        Returns:
            bool: True if connection is successful, False otherwise
        """
        try:
            response = requests.get(
                f"{self.server_url}/health",
                headers=self.headers,
                timeout=5
            )
            if response.status_code == 200:
                print(f"✅ Successfully connected to MCP server at {self.server_url}")
                return True
            else:
                print(f"❌ Failed to connect to MCP server. Status code: {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            print(f"❌ Error connecting to MCP server: {e}")
            return False

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

        except requests.exceptions.HTTPError as e:
            print(f"❌ HTTP Error: {e}")
            # Try to parse error response
            try:
                error_data = response.json()
                print(f"Server error details: {error_data}")
                return {"error": error_data}
            except:
                return {"error": str(e)}

        except requests.exceptions.ConnectionError as e:
            print(f"❌ Connection Error: {e}")
            return {"error": "Failed to connect to the server"}

        except requests.exceptions.Timeout as e:
            print(f"❌ Timeout Error: {e}")
            return {"error": "Request timed out"}

        except requests.exceptions.RequestException as e:
            print(f"❌ Request Error: {e}")
            return {"error": str(e)}

    def get_models(self) -> List[Dict[str, Any]]:
        """
        Get a list of available models from the MCP server.

        Returns:
            List of model information dictionaries
        """
        response = self.send_request("models", {}, method="GET")
        if "error" in response:
            print("Failed to retrieve models")
            return []
        return response.get("models", [])

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

    def batch_predict(self, model_name: str, batch_inputs: List[Dict[str, Any]],
                      parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Send a batch prediction request to the MCP server.

        Args:
            model_name: Name of the model to use for prediction
            batch_inputs: List of input data for the model
            parameters: Optional parameters for the prediction

        Returns:
            List of dictionaries containing the prediction results
        """
        data = {
            "model": model_name,
            "batch_inputs": batch_inputs
        }

        if parameters:
            data["parameters"] = parameters

        response = self.send_request("batch_predict", data)
        return response.get("results", [])

    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific model.

        Args:
            model_name: Name of the model

        Returns:
            Dict containing model information
        """
        return self.send_request(f"models/{model_name}", {}, method="GET")


# Example usage
def main():
    """
    Example demonstrating how to use the MCPClient class.
    """
    print("=" * 50)
    print("MCP Client Example")
    print("=" * 50)

    # Initialize client
    print("\n1. Initializing MCP Client...")
    client = MCPClient(server_url="http://localhost:8000", api_key="your_api_key_here")

    # Get available models
    print("\n2. Getting available models...")
    models = client.get_models()
    print(f"Available models: {json.dumps(models, indent=2)}")

    # Get model information
    print("\n3. Getting information about a specific model...")
    model_info = client.get_model_info("text-classification")
    print(f"Model info: {json.dumps(model_info, indent=2)}")

    # Make a prediction
    print("\n4. Making a prediction...")
    prediction_input = {
        "text": "This movie was fantastic! I really enjoyed it."
    }
    prediction_params = {
        "temperature": 0.7,
        "max_length": 100
    }

    prediction_result = client.predict(
        model_name="text-classification",
        inputs=prediction_input,
        parameters=prediction_params
    )
    print(f"Prediction result: {json.dumps(prediction_result, indent=2)}")

    # Batch prediction
    print("\n5. Making a batch prediction...")
    batch_inputs = [
        {"text": "I loved this product!"},
        {"text": "The service was terrible."},
        {"text": "It was okay, nothing special."}
    ]

    batch_results = client.batch_predict(
        model_name="text-classification",
        batch_inputs=batch_inputs,
        parameters={"temperature": 0.5}
    )
    print(f"Batch prediction results: {json.dumps(batch_results, indent=2)}")

    print("\n" + "=" * 50)
    print("Example completed!")
    print("=" * 50)


# Advanced example with error handling and retries
def advanced_example():
    """
    Advanced example demonstrating error handling and retries.
    """
    print("\n\n" + "=" * 50)
    print("Advanced MCP Client Example")
    print("=" * 50)

    # Initialize client with error handling
    print("\n1. Initializing MCP Client with error handling...")
    try:
        client = MCPClient(server_url="http://localhost:8000", api_key="your_api_key_here")
    except Exception as e:
        print(f"Failed to initialize client: {e}")
        return

    # Example with retries
    print("\n2. Making a prediction with retries...")
    max_retries = 3
    retry_delay = 2  # seconds

    prediction_input = {
        "text": "What is the capital of France?"
    }

    for attempt in range(max_retries):
        try:
            print(f"Attempt {attempt + 1}/{max_retries}...")
            result = client.predict(
                model_name="question-answering",
                inputs=prediction_input
            )

            if "error" not in result:
                print(f"Success! Result: {json.dumps(result, indent=2)}")
                break
            else:
                print(f"Error in response: {result['error']}")

            if attempt < max_retries - 1:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                # Exponential backoff
                retry_delay *= 2
        except Exception as e:
            print(f"Exception during prediction: {e}")
            if attempt < max_retries - 1:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                # Exponential backoff
                retry_delay *= 2

    print("\n" + "=" * 50)
    print("Advanced example completed!")
    print("=" * 50)


if __name__ == "__main__":
    # Run the basic example
    main()

    # Run the advanced example
    advanced_example()
