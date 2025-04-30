| Rank | Project / Example                                 | Key new concepts that make it harder than the one above                                                                                                   |
|------|---------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1    | **Basic / Echo + Calculator MCP Server**          | First look at `FastMCP`, single-file, two toy tools, runs over stdio.                                                                                     |
| 2    | **User Profile MCP Server**                       | Adds a dynamic **Resource** path and a second domain object.                                                                                              |
| 3    | **Simple MCP Client**                             | Shows `ClientSession` & `stdio_client`, but no server logic.                                                                                              |
| 4    | **Code Review Helper MCP Server**                 | Introduces the `@mcp.prompt` decorator; still “string-in, string-out”.                                                                                    |
| 5    | **Weather-Forecast Tool**                         | First **async HTTP** call and JSON parsing, plus 404 handling.                                                                                            |
| 6    | **Multi-Tool LangGraph Research Agent**           | Combines *several* MCP servers, conditional branches, planning node, and result synthesis in LangGraph—hardest because it mixes everything above at once. |
| 7    | **Secure File-Reader**                            | Path-sanitisation (root locking) and file-size truncation.                                                                                                |
| 22   | **Multi-Server, Multi-Step Research Agent**       | Coordinates **two** running MCP servers inside one LangChain agent.                                                                                       |
| 8    | **CSV Quick-Insights Tool**                       | Brings in **pandas** for I/O and summarisation.                                                                                                           |
| 15   | **LangChain Agent with MCP Calculator Tool**      | Bridges MCP → LangChain; need tool adapters & agent loop.                                                                                                 |
| 23   | **HTTP-SSE Deployment & Streaming Resource**      | Network transport change, chunked streaming, remote client connect.                                                                                       |
| 8    | **Chatbot Context Server & Client**               | Resources feed an external LLM; requires coordinating two processes.                                                                                      |
| 9    | **Ollama Client with MCP Context**                | Adds a local LLM REST call and prompt templating.                                                                                                         |
| 10   | **Stock-Price + Sentiment Bundler**               | Aggregates **two** APIs and does basic NLP polarity scoring.                                                                                              |
| 11   | **PostgreSQL Query Server**                       | Real DB pool (`asyncpg`) and a read-only SQL guard with regex.                                                                                            |
| 12   | **Notion Page Creator**                           | First **write-back** workflow; OAuth-style secret management.                                                                                             |
| 13   | **Secure MCP Server (API-Key Auth)**              | Adds explicit request-level auth & error handling paths.                                                                                                  |
| 14   | **Production-Ready MCP Server (Lifespan Mgmt)**   | Requires startup/shutdown hooks and resource cleanup.                                                                                                     |
| 16   | **RAG Enhancement with MCP Real-time Data**       | Orchestrates vector store retrieval *and* live MCP context.                                                                                               |
| 17   | **Simple LangGraph Workflow Calling an MCP Tool** | First LangGraph state machine; tool call embedded in a node.                                                                                              |
| 19   | **Local Llama 2 Agent (Ollama + LangChain)**      | Offline LLM, larger model weights, plus LangChain tool harness.                                                                                           |
| 20   | **PR Assistant End-to-End (Client + Server)**     | Spawns the server as a subprocess and mediates LLM ↔ tools loops.                                                                                         |
| 21   | **LangGraph with External MCP State Management**  | Two-way state read/write over MCP inside a multi-node graph.                                                                                              |

Okay, let's break down the coding outline for each MCP *server* component based on your ranked list. We'll focus on the
server-side logic, assuming `FastMCP` is the core framework.

**Core Assumption:** `FastMCP` provides decorators like `@mcp.tool`, `@mcp.resource`, `@mcp.prompt`, and a way to run
the server (e.g., `mcp.run_stdio()`, `mcp.run_http()`). We'll infer standard Python practices using `asyncio` for I/O
bound tasks.

---

### Basic / Echo + Calculator MCP Server

* **Objective:** Provide simple 'echo' and 'calculator' tools over standard I/O.
* **Key Concepts:** `FastMCP` basics, single-file structure, stdio transport, defining simple tools.
* **Core Implementation Details:**
    * **Framework:** `FastMCP`
    * **File Structure:** Single Python file (e.g., `basic_server.py`).
    * **Main Components:**
        * Import `FastMCP`.
        * Instantiate the MCP app: `mcp = FastMCP()`.
        * Define an `echo` tool using `@mcp.tool`:
          ```python
          @mcp.tool("echo")
          async def echo_tool(text: str) -> str:
              """Returns the input text unchanged."""
              return text
          ```
        * Define a `calculator` tool using `@mcp.tool`:
          ```python
          import operator

          OPS = {
              "+": operator.add, "-": operator.sub,
              "*": operator.mul, "/": operator.truediv,
          }

          @mcp.tool("calculate")
          async def calculate_tool(operand1: float, operator_str: str, operand2: float) -> float:
              """Performs a basic arithmetic operation."""
              if operator_str not in OPS:
                  raise ValueError(f"Unknown operator: {operator_str}")
              if operator_str == '/' and operand2 == 0:
                  raise ValueError("Division by zero")
              return OPS[operator_str](operand1, operand2)
          ```
        * Add a main execution block:
          ```python
          if __name__ == "__main__":
              mcp.run_stdio() # Or appropriate run function
          ```
    * **Key Libraries:** `FastMCP`, `asyncio` (implicitly used by FastMCP), standard `operator` module.
    * **Data Handling:** Simple types (string, float). `FastMCP` likely handles basic type conversion/validation.
    * **Error Handling:** Basic `ValueError` for invalid calculator input.
    * **Deployment:** Runs via standard input/output using `mcp.run_stdio()`.

---

### User Profile MCP Server

* **Objective:** Manage user profile data using dynamic resource paths.
* **Key Concepts:** Dynamic `Resource` paths, handling domain objects (User Profiles).
* **Core Implementation Details:**
    * **Framework:** `FastMCP`
    * **File Structure:** Likely still a single file, possibly defining a Pydantic model for the User Profile.
    * **Main Components:**
        * `mcp = FastMCP()`
        * Define a data model (e.g., using `pydantic`):
          ```python
          from pydantic import BaseModel
          from typing import Optional

          class UserProfile(BaseModel):
              user_id: str
              name: str
              email: Optional[str] = None
              preferences: dict = {}
          ```
        * Use an in-memory dictionary for storage (simplest case):
          `user_profiles_db = {}`
        * Define a Resource using `@mcp.resource`:
          ```python
          @mcp.resource("/users/{user_id}")
          class UserProfileResource:
              async def get(self, user_id: str) -> UserProfile:
                  """Retrieves a user profile."""
                  if user_id not in user_profiles_db:
                      raise ValueError(f"User {user_id} not found") # Or specific MCP not found error
                  return user_profiles_db[user_id]

              async def put(self, user_id: str, profile_data: UserProfile) -> UserProfile:
                  """Creates or updates a user profile."""
                  if user_id != profile_data.user_id:
                       raise ValueError("User ID mismatch in path and body")
                  user_profiles_db[user_id] = profile_data
                  return profile_data

              # Optional: Add delete method
              async def delete(self, user_id: str) -> dict:
                   if user_id not in user_profiles_db:
                       raise ValueError(f"User {user_id} not found")
                   del user_profiles_db[user_id]
                   return {"status": "deleted", "user_id": user_id}

          ```
    * **Key Libraries:** `FastMCP`, `asyncio`, `pydantic`.
    * **Data Handling:** Uses Pydantic models for structured data and validation. `FastMCP` integrates with the resource
      methods (GET, PUT, DELETE).
    * **Error Handling:** Handles 'not found' cases, potential validation errors via Pydantic.
    * **Deployment:** Still likely `stdio` at this stage, but the resource structure anticipates other transports.

---

### Code Review Helper MCP Server

* **Objective:** Use an LLM prompt (via `@mcp.prompt`) to provide code review suggestions.
* **Key Concepts:** `@mcp.prompt` decorator, potentially integrating with an external LLM service.
* **Core Implementation Details:**
    * **Framework:** `FastMCP`
    * **File Structure:** Single file.
    * **Main Components:**
        * `mcp = FastMCP()`
        * Define the prompt template (this might be configured in the decorator or globally).
        * Use the `@mcp.prompt` decorator. *Assumption:* This decorator might abstract the LLM call or simply format the
          input for a tool function that then makes the call. Let's assume the latter for more control.
          ```python
          import httpx # To call an external LLM API

          # Simplified example - real implementation needs API key, error handling etc.
          LLM_API_URL = "http://localhost:11434/api/generate" # Example: Ollama

          PROMPT_TEMPLATE = """
          Review the following Python code and provide suggestions for improvement:
          ```python
          {code_snippet}
          ```
          Suggestions:
          """

          @mcp.tool("review_code") # Or potentially @mcp.prompt(...) if it handles the call
          async def review_code_tool(code_snippet: str) -> str:
          """Analyzes code using an LLM and returns review suggestions."""
          prompt = PROMPT_TEMPLATE.format(code_snippet=code_snippet)
          try:
          # Example using Ollama API structure
          async with httpx.AsyncClient() as client:
          response = await client.post(
          LLM_API_URL,
          json={"model": "llama3", "prompt": prompt, "stream": False},
          timeout=60.0
          )
          response.raise_for_status() # Raise HTTP errors
          # Assuming response format provides the text directly or in a specific key
          return response.json().get("response", "Error: No response text found")
          except httpx.HTTPStatusError as e:
          return f"Error calling LLM: {e.response.status_code} - {e.response.text}"
          except Exception as e:
          return f"An unexpected error occurred: {e}"
          ```
        * *Alternative:* If `@mcp.prompt` *does* handle the LLM call, the function might be simpler:
          ```python
          # This assumes @mcp.prompt handles the LLM interaction based on the template
          # @mcp.prompt(template=PROMPT_TEMPLATE, llm_config={...})
          # async def review_code_prompt(code_snippet: str) -> str:
          #    # The decorator might inject the LLM response here, or the function
          #    # might just need to return the input for the decorator to process.
          #    # This depends heavily on FastMCP's specific design for @mcp.prompt.
          #    # Let's stick to the explicit call for clarity unless specified otherwise.
          #    pass
          ```
    * **Key Libraries:** `FastMCP`, `asyncio`, `httpx` (or `aiohttp`).
    * **Data Handling:** String input (code), string output (review). JSON for the LLM API call.
    * **Error Handling:** Includes HTTP error handling for the LLM call, plus general exceptions.
    * **Considerations:** LLM API key management, prompt engineering, LLM response parsing.

---

### Weather-Forecast Tool (Server Component)

* **Objective:** Provide a tool that fetches weather forecasts from an external HTTP API.
* **Key Concepts:** Async HTTP calls, JSON parsing, external API error handling (e.g., 404 Not Found).
* **Core Implementation Details:**
    * **Framework:** `FastMCP`
    * **File Structure:** Single file.
    * **Main Components:**
        * `mcp = FastMCP()`
        * Need an API key for a weather service (e.g., OpenWeatherMap). Load securely (env var).
        * Define the tool:
          ```python
          import httpx
          import os

          WEATHER_API_KEY = os.environ.get("OPENWEATHERMAP_API_KEY")
          WEATHER_API_URL = "https://api.openweathermap.org/data/2.5/weather"

          @mcp.tool("get_weather")
          async def get_weather_tool(location: str) -> dict:
               """Fetches the current weather for a given location."""
               if not WEATHER_API_KEY:
                   raise ValueError("Weather API key not configured")

               params = {"q": location, "appid": WEATHER_API_KEY, "units": "metric"}
               async with httpx.AsyncClient() as client:
                   try:
                       response = await client.get(WEATHER_API_URL, params=params)
                       response.raise_for_status() # Raises exception for 4xx/5xx responses
                       return response.json()
                   except httpx.HTTPStatusError as e:
                       if e.response.status_code == 404:
                           raise ValueError(f"Location '{location}' not found.") from e
                       else:
                           raise RuntimeError(f"Weather API error: {e.response.status_code}") from e
                   except Exception as e:
                       raise RuntimeError(f"Failed to fetch weather: {e}") from e
          ```
    * **Key Libraries:** `FastMCP`, `asyncio`, `httpx`, `os` (for API key).
    * **Data Handling:** String input (location), dictionary/JSON output (weather data). Parses JSON response.
    * **Error Handling:** Explicitly checks for 404, handles other HTTP errors, checks for missing API key. Uses
      `RuntimeError` for unexpected issues.
    * **Security:** API key management is crucial.

---

### Secure File-Reader (Server Component)

* **Objective:** Provide a tool to securely read files from a designated root directory.
* **Key Concepts:** Path sanitisation, preventing directory traversal (root locking), file size limits.
* **Core Implementation Details:**
    * **Framework:** `FastMCP`
    * **File Structure:** Single file.
    * **Main Components:**
        * `mcp = FastMCP()`
        * Define a secure base directory. Load from config/env var.
        * Define the tool:
          ```python
          import os
          import aiofiles # For async file operations

          # Define a secure root directory where files can be read from
          # IMPORTANT: This should be an absolute path configured securely.
          SECURE_ROOT_DIR = os.path.abspath("./readable_files")
          MAX_FILE_SIZE_BYTES = 1024 * 1024 # 1MB limit example

          @mcp.tool("read_secure_file")
          async def read_secure_file_tool(relative_path: str) -> str:
              """Reads a file securely from a predefined directory."""

              if not os.path.exists(SECURE_ROOT_DIR):
                   os.makedirs(SECURE_ROOT_DIR, exist_ok=True) # Create if not exists for demo

              # Attempt to construct the full path
              try:
                  # Normalize the path (e.g., collapses '..')
                  norm_path = os.path.normpath(relative_path)
                  # Prevent path starting with '/', '\', or containing '..' components after normalization
                  if norm_path.startswith(('..', '/', '\\')) or '..' in norm_path.split(os.sep):
                       raise ValueError("Invalid relative path provided.")

                  full_path = os.path.join(SECURE_ROOT_DIR, norm_path)
                  abs_full_path = os.path.abspath(full_path)

                  # **Crucial Security Check:** Ensure the resolved path is still within the root dir
                  if not abs_full_path.startswith(SECURE_ROOT_DIR):
                      raise ValueError("Path traversal attempt detected.")

                  # Check file existence and type
                  if not os.path.isfile(abs_full_path):
                      raise FileNotFoundError(f"File not found or is not a regular file: {relative_path}")

                  # Check file size before reading
                  file_size = os.path.getsize(abs_full_path)
                  if file_size > MAX_FILE_SIZE_BYTES:
                      raise ValueError(f"File exceeds maximum size limit of {MAX_FILE_SIZE_BYTES} bytes.")

                  # Read the file asynchronously
                  async with aiofiles.open(abs_full_path, mode='r', encoding='utf-8') as f:
                      content = await f.read()
                  return content

              except FileNotFoundError as e:
                  raise FileNotFoundError(str(e)) # Re-raise for clarity
              except ValueError as e:
                  raise ValueError(str(e)) # Re-raise security/validation errors
              except Exception as e:
                  # Log the exception details here
                  raise RuntimeError(f"An error occurred reading the file: {e}") from e
          ```
    * **Key Libraries:** `FastMCP`, `asyncio`, `os`, `aiofiles`.
    * **Data Handling:** String input (relative path), string output (file content).
    * **Error Handling:** Catches `FileNotFoundError`, permission errors (implicitly via `aiofiles.open`), size limit
      errors, and path validation errors.
    * **Security:** This is the core focus. Uses `os.path.abspath`, `os.path.normpath`, checks `startswith`, and
      validates against `..` components to prevent escaping the `SECURE_ROOT_DIR`. File size limit prevents DoS. *
      *Crucial:** The `SECURE_ROOT_DIR` must be set correctly.

---

### CSV Quick-Insights Tool (Server Component)

* **Objective:** Provide a tool to load a CSV and return summary statistics using pandas.
* **Key Concepts:** Using `pandas` for CSV I/O and data analysis.
* **Core Implementation Details:**
    * **Framework:** `FastMCP`
    * **File Structure:** Single file.
    * **Main Components:**
        * `mcp = FastMCP()`
        * *Option 1: Input is file path (Leverage Secure File Reader)*
            * This tool could *call* the `read_secure_file_tool` internally or reuse its secure path logic.
        * *Option 2: Input is raw CSV content as a string* (Simpler for this example)
        * Define the tool:
          ```python
          import pandas as pd
          from io import StringIO
          import asyncio

          @mcp.tool("csv_insights")
          async def csv_insights_tool(csv_content: str) -> dict:
              """Generates basic insights from CSV data using pandas."""
              if not csv_content:
                  raise ValueError("CSV content cannot be empty.")

              try:
                  # Use StringIO to treat the string as a file
                  csv_file = StringIO(csv_content)

                  # Run pandas potentially blocking operations in a thread pool executor
                  # to avoid blocking the asyncio event loop.
                  loop = asyncio.get_running_loop()
                  df = await loop.run_in_executor(None, pd.read_csv, csv_file)

                  # Generate insights (run these in executor too if potentially slow)
                  info_str = await loop.run_in_executor(None, df.info, {"buf": StringIO()})
                  info_output = info_str.getvalue() if info_str else "Could not get df.info()"


                  describe_df = await loop.run_in_executor(None, df.describe)
                  head_df = await loop.run_in_executor(None, df.head)

                  insights = {
                      "shape": df.shape,
                      "columns": df.columns.tolist(),
                      "info": info_output, # df.info() prints to buffer
                      "description": describe_df.to_dict(),
                      "head": head_df.to_dict(orient='records'),
                      # Add more insights as needed (e.g., missing values)
                      "missing_values": df.isnull().sum().to_dict()
                  }
                  return insights

              except pd.errors.EmptyDataError:
                  raise ValueError("Provided CSV content is empty or invalid.") from None
              except Exception as e:
                  raise RuntimeError(f"Error processing CSV: {e}") from e
          ```
    * **Key Libraries:** `FastMCP`, `asyncio`, `pandas`, `io`.
    * **Data Handling:** String input (CSV data), dictionary/JSON output (insights). Uses `StringIO` to handle string
      input for pandas.
    * **Performance:** Uses `asyncio.get_running_loop().run_in_executor` to run potentially blocking `pandas` operations
      in a separate thread, preventing blocking of the main async event loop. This is important for larger CSV files.
    * **Error Handling:** Catches pandas-specific errors like `EmptyDataError` and general exceptions.

---

### Chatbot Context Server (Server Component)

* **Objective:** Manage conversation history for different chat sessions, potentially feeding context to an external
  LLM.
* **Key Concepts:** Storing/retrieving structured conversational data, likely using dynamic resources keyed by session
  ID.
* **Core Implementation Details:**
    * **Framework:** `FastMCP`
    * **File Structure:** Single file, likely with a Pydantic model for message structure.
    * **Main Components:**
        * `mcp = FastMCP()`
        * Define a message model:
          ```python
          from pydantic import BaseModel
          from typing import List, Dict, Any
          import datetime

          class ChatMessage(BaseModel):
              role: str # e.g., 'user', 'assistant', 'system'
              content: str
              timestamp: datetime.datetime = datetime.datetime.now(datetime.timezone.utc)
              metadata: Dict[str, Any] = {}
          ```
        * Use an in-memory dictionary keyed by session ID to store conversations:
          `chat_histories: Dict[str, List[ChatMessage]] = {}`
        * Define a Resource using `@mcp.resource`:
          ```python
          @mcp.resource("/chats/{session_id}/history")
          class ChatHistoryResource:
              async def get(self, session_id: str) -> List[ChatMessage]:
                  """Retrieves the full chat history for a session."""
                  if session_id not in chat_histories:
                      # Return empty list if session doesn't exist yet
                      return []
                  return chat_histories[session_id]

              async def post(self, session_id: str, message: ChatMessage) -> List[ChatMessage]:
                  """Appends a new message to the chat history."""
                  if session_id not in chat_histories:
                      chat_histories[session_id] = []
                  chat_histories[session_id].append(message)
                  # Optional: Implement history truncation/summarization logic here
                  return chat_histories[session_id] # Return updated history

              # Optional: Add a DELETE method to clear a history
              async def delete(self, session_id: str) -> dict:
                  if session_id in chat_histories:
                      del chat_histories[session_id]
                      return {"status": "deleted", "session_id": session_id}
                  else:
                       # Or raise not found error
                      return {"status": "not_found", "session_id": session_id}
          ```
        * Add a tool to get formatted context (e.g., for an LLM prompt):
          ```python
          @mcp.tool("get_formatted_chat_context")
          async def get_formatted_context(session_id: str, max_messages: int = 10) -> str:
              """Retrieves and formats the recent chat history for LLM input."""
              history = chat_histories.get(session_id, [])
              # Get the last 'max_messages' messages
              recent_history = history[-max_messages:]
              # Format appropriately (example)
              formatted = "\n".join([f"{msg.role.capitalize()}: {msg.content}" for msg in recent_history])
              return formatted
          ```
    * **Key Libraries:** `FastMCP`, `asyncio`, `pydantic`, `datetime`.
    * **Data Handling:** Uses Pydantic models for messages. Stores lists of messages in a dictionary. JSON input/output
      for the resource.
    * **Error Handling:** Handles non-existent sessions gracefully (e.g., returning empty lists or specific statuses).
      Potential validation errors via Pydantic.
    * **Considerations:** In-memory storage is volatile. For persistence, replace the dictionary with a database
      connection (async). Implement logic for history length management (truncation, summarization).

---

### Stock-Price + Sentiment Bundler (Server Component)

* **Objective:** Aggregate stock price data from one API and related news sentiment from another (or same, if
  available), returning a combined result.
* **Key Concepts:** Calling multiple async HTTP APIs concurrently, basic NLP for sentiment scoring (if needed),
  combining results.
* **Core Implementation Details:**
    * **Framework:** `FastMCP`
    * **File Structure:** Single file. Might include a simple sentiment analysis utility function or library.
    * **Main Components:**
        * `mcp = FastMCP()`
        * API keys for stock data (e.g., Alpha Vantage, Finnhub) and potentially news (e.g., NewsAPI).
        * Define the tool:
          ```python
          import httpx
          import asyncio
          import os
          # Optional: basic sentiment analysis (example using a simple lexicon)
          # from some_sentiment_analyzer import simple_polarity

          STOCK_API_KEY = os.environ.get("STOCK_API_KEY")
          STOCK_API_URL = "https://www.alphavantage.co/query" # Example
          NEWS_API_KEY = os.environ.get("NEWS_API_KEY")
          NEWS_API_URL = "https://newsapi.org/v2/everything" # Example

          async def fetch_stock_price(client: httpx.AsyncClient, symbol: str) -> dict:
              """Fetches stock quote."""
              params = {
                  "function": "GLOBAL_QUOTE",
                  "symbol": symbol,
                  "apikey": STOCK_API_KEY
              }
              try:
                  response = await client.get(STOCK_API_URL, params=params)
                  response.raise_for_status()
                  data = response.json()
                  # Extract relevant price info - structure depends on API
                  quote = data.get("Global Quote", {})
                  return {
                      "symbol": quote.get("01. symbol"),
                      "price": quote.get("05. price"),
                      "change_percent": quote.get("10. change percent")
                  }
              except Exception as e:
                  # Log error
                  return {"error": f"Failed to fetch stock price: {e}"}

          async def fetch_news_sentiment(client: httpx.AsyncClient, query: str) -> dict:
              """Fetches news headlines and calculates simple average sentiment."""
              params = {
                  "q": query, # Search query, e.g., company name or stock symbol
                  "apiKey": NEWS_API_KEY,
                  "language": "en",
                  "pageSize": 5 # Limit number of articles
              }
              try:
                  response = await client.get(NEWS_API_URL, params=params)
                  response.raise_for_status()
                  data = response.json()
                  articles = data.get("articles", [])
                  if not articles:
                      return {"sentiment_score": 0, "headline_count": 0, "headlines": []}

                  # Example: Simple sentiment (replace with actual NLP if needed)
                  sentiment_scores = []
                  headlines = []
                  for article in articles:
                      title = article.get("title", "")
                      headlines.append(title)
                      # score = simple_polarity(title + " " + article.get("description",""))
                      # sentiment_scores.append(score) # Placeholder

                  # avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
                  avg_sentiment = 0 # Placeholder for actual calculation

                  return {
                      "sentiment_score": avg_sentiment, # Placeholder
                      "headline_count": len(articles),
                      "headlines": headlines[:3] # Return a few example headlines
                  }
              except Exception as e:
                  # Log error
                  return {"error": f"Failed to fetch news/sentiment: {e}"}


          @mcp.tool("get_stock_and_sentiment")
          async def stock_sentiment_tool(symbol: str, company_name: str) -> dict:
              """Fetches stock price and related news sentiment."""
              if not STOCK_API_KEY or not NEWS_API_KEY:
                   raise ValueError("API keys for stock and/or news are not configured.")

              async with httpx.AsyncClient() as client:
                  # Run API calls concurrently
                  stock_task = fetch_stock_price(client, symbol)
                  sentiment_task = fetch_news_sentiment(client, company_name) # Use company name for news query

                  stock_result, sentiment_result = await asyncio.gather(
                      stock_task,
                      sentiment_task,
                      return_exceptions=True # Allow one to fail without stopping the other
                  )

                  # Handle potential errors returned from gather
                  if isinstance(stock_result, Exception):
                      stock_data = {"error": str(stock_result)}
                  else:
                      stock_data = stock_result

                  if isinstance(sentiment_result, Exception):
                      sentiment_data = {"error": str(sentiment_result)}
                  else:
                      sentiment_data = sentiment_result

                  return {
                      "stock_data": stock_data,
                      "sentiment_data": sentiment_data
                  }
          ```
    * **Key Libraries:** `FastMCP`, `asyncio`, `httpx`, `os`. Maybe a basic NLP library if doing real sentiment.
    * **Data Handling:** String inputs (symbol, company name). Fetches JSON from two APIs. Combines results into a
      single dictionary/JSON output.
    * **Error Handling:** Handles HTTP errors for *each* API call independently using
      `asyncio.gather(..., return_exceptions=True)`. Checks for missing API keys.
    * **Performance:** Uses `asyncio.gather` for concurrent API calls, speeding up the process.
    * **Security:** API key management. Be mindful of API rate limits.

---

### PostgreSQL Query Server

* **Objective:** Allow executing secure, read-only SQL queries against a PostgreSQL database.
* **Key Concepts:** Database connection pooling (`asyncpg`), preventing SQL injection (parameterization), enforcing
  read-only access (regex guard or user permissions).
* **Core Implementation Details:**
    * **Framework:** `FastMCP`
    * **File Structure:** Single file, potentially needing environment variables for DB connection.
    * **Main Components:**
        * `mcp = FastMCP()`
        * Database connection pool (managed via lifespan events - see #14, but shown here for completeness).
        * Read-only SQL check function.
        * Define the tool:
          ```python
          import asyncpg
          import os
          import re
          from typing import List, Dict, Any

          DB_POOL = None # Global pool variable, initialized during startup

          # Basic regex to block modifying statements (improve as needed)
          # WARNING: Regex is NOT a foolproof way to prevent all malicious SQL.
          #          Database user permissions are the most reliable method.
          READ_ONLY_REGEX = re.compile(r"^\s*(SELECT|WITH)\s+", re.IGNORECASE)

          async def startup_db():
              global DB_POOL
              try:
                  DB_POOL = await asyncpg.create_pool(
                      user=os.environ.get('PG_USER'),
                      password=os.environ.get('PG_PASSWORD'),
                      database=os.environ.get('PG_DATABASE'),
                      host=os.environ.get('PG_HOST', 'localhost'),
                      port=os.environ.get('PG_PORT', 5432)
                  )
                  print("Database pool created.")
              except Exception as e:
                  print(f"Failed to create database pool: {e}")
                  # Decide how to handle this - exit, retry?
                  DB_POOL = None

          async def shutdown_db():
              global DB_POOL
              if DB_POOL:
                  await DB_POOL.close()
                  print("Database pool closed.")

          # Register lifespan events (assuming FastMCP supports this)
          # mcp.add_event_handler("startup", startup_db)
          # mcp.add_event_handler("shutdown", shutdown_db)
          # If not, pool creation/closing needs manual trigger or context manager approach

          @mcp.tool("execute_sql_readonly")
          async def execute_sql_tool(query: str, params: List[Any] = None) -> List[Dict[str, Any]]:
              """Executes a read-only SQL query with parameters."""
              if not DB_POOL:
                   raise RuntimeError("Database connection pool is not available.")

              # Security Check 1: Basic statement type check (use with caution)
              if not READ_ONLY_REGEX.match(query):
                  raise ValueError("Only SELECT or WITH statements are allowed.")

              # Security Check 2: Use parameterization (asyncpg handles this)
              if params is None:
                  params = []

              try:
                  async with DB_POOL.acquire() as connection:
                      # Security Check 3: Ideally, the DB user connected should ONLY have SELECT permissions.
                      # The `fetch` method executes the query.
                      results = await connection.fetch(query, *params)
                      # Convert asyncpg Records to dictionaries for easier JSON serialization
                      return [dict(record) for record in results]
              except asyncpg.exceptions.PostgresSyntaxError as e:
                  raise ValueError(f"SQL Syntax Error: {e}") from e
              except Exception as e:
                  # Log the detailed error
                  raise RuntimeError(f"Database query failed: {e}") from e
          ```
    * **Key Libraries:** `FastMCP`, `asyncio`, `asyncpg`, `os`, `re`.
    * **Data Handling:** String input (SQL query), list input (parameters). Returns list of dictionaries (rows).
      `asyncpg` handles type conversions between Python and PostgreSQL.
    * **Error Handling:** Catches `asyncpg` specific errors (like syntax errors), general database exceptions, pool
      availability issues
      Okay, continuing with the remaining MCP server outlines, using headers for the project/example titles:

---

### Notion Page Creator

* **Objective:** Create a new page in Notion based on provided content using the Notion API.
* **Key Concepts:** Write operation via external API, OAuth or API token management, constructing API-specific payload (
  Notion Block Kit).
* **Core Implementation Details:**
    * **Framework:** `FastMCP`
    * **File Structure:** Single file.
    * **Main Components:**
        * `mcp = FastMCP()`
        * Notion API Integration Token (Internal Integration Token recommended).
        * Parent Page ID where new pages will be created.
        * Define the tool:
          ```python
          import httpx
          import os
          from typing import Dict, Any

          NOTION_API_KEY = os.environ.get("NOTION_API_KEY")
          NOTION_API_VERSION = "2022-06-28" # Use the target API version
          NOTION_BASE_URL = "https://api.notion.com/v1"
          # ID of the parent page/database under which new pages will be created
          NOTION_PARENT_PAGE_ID = os.environ.get("NOTION_PARENT_PAGE_ID")

          @mcp.tool("create_notion_page")
          async def create_notion_page_tool(title: str, content_markdown: str) -> Dict[str, Any]:
              """Creates a new page in Notion under a predefined parent page."""
              if not NOTION_API_KEY:
                  raise ValueError("Notion API Key not configured.")
              if not NOTION_PARENT_PAGE_ID:
                  raise ValueError("Notion Parent Page ID not configured.")

              headers = {
                  "Authorization": f"Bearer {NOTION_API_KEY}",
                  "Content-Type": "application/json",
                  "Notion-Version": NOTION_API_VERSION,
              }

              # Basic conversion from Markdown to Notion blocks (very simplified)
              # A real implementation might use a library or more sophisticated parsing.
              blocks = []
              paragraphs = content_markdown.split('\n\n') # Split by double newline
              for para in paragraphs:
                  if para.strip(): # Ignore empty paragraphs
                       blocks.append({
                           "object": "block",
                           "type": "paragraph",
                           "paragraph": {
                               "rich_text": [{"type": "text", "text": {"content": para.strip()}}]
                           }
                       })

              # Construct the API payload
              payload = {
                  "parent": {"page_id": NOTION_PARENT_PAGE_ID},
                  "properties": {
                      # Assumes the parent page has a standard 'title' property
                      "title": {
                          "title": [{"type": "text", "text": {"content": title}}]
                      }
                  },
                  "children": blocks # Add the content blocks
              }

              async with httpx.AsyncClient() as client:
                  try:
                      response = await client.post(
                          f"{NOTION_BASE_URL}/pages",
                          headers=headers,
                          json=payload
                      )
                      response.raise_for_status() # Check for API errors
                      result_data = response.json()
                      return {
                           "status": "success",
                           "page_id": result_data.get("id"),
                           "url": result_data.get("url")
                      }
                  except httpx.HTTPStatusError as e:
                      error_details = e.response.json()
                      raise RuntimeError(f"Notion API Error: {e.response.status_code} - {error_details.get('message', e.response.text)}") from e
                  except Exception as e:
                      raise RuntimeError(f"Failed to create Notion page: {e}") from e

          ```
    * **Key Libraries:** `FastMCP`, `asyncio`, `httpx`, `os`.
    * **Data Handling:** String inputs (title, markdown content). Constructs a complex JSON payload specific to the
      Notion API. Returns a dictionary with page ID/URL.
    * **Error Handling:** Handles missing configuration (API key, parent ID), HTTP errors from the Notion API (parsing
      error messages if possible), and general exceptions.
    * **Security:** Securely manage the Notion API token. Ensure the token has the minimum required permissions (write
      access to the target parent page).
    * **Complexity:** Mapping input content (like Markdown) to Notion's block structure can be complex and may require a
      dedicated parsing function or library for robust handling.

---

### Secure MCP Server (API-Key Auth)

* **Objective:** Add request-level authentication using API keys.
* **Key Concepts:** Middleware or decorator-based authentication, validating API keys against a store, handling
  authentication errors.
* **Core Implementation Details:**
    * **Framework:** `FastMCP` (*Assumption: FastMCP provides a middleware or decorator mechanism similar to
      FastAPI/Starlette*).
    * **File Structure:** Single file, or potentially auth logic in a separate module.
    * **Main Components:**
        * `mcp = FastMCP()`
        * A way to store valid API keys (e.g., environment variables, a simple set, a database).
        * Authentication middleware or decorator function.
          ```python
          import os
          from typing import Optional

          # Example: Load allowed keys from an environment variable (comma-separated)
          ALLOWED_API_KEYS_STR = os.environ.get("ALLOWED_API_KEYS", "")
          VALID_API_KEYS = set(key.strip() for key in ALLOWED_API_KEYS_STR.split(',') if key.strip())

          # --- Option 1: Middleware (Conceptual - depends on FastMCP implementation) ---
          # async def api_key_middleware(request: MCPRequest, call_next): # MCPRequest is hypothetical
          #     api_key = request.headers.get("X-API-Key") # Or get from query param, etc.
          #
          #     if not api_key or api_key not in VALID_API_KEYS:
          #         # Return an unauthorized response (structure depends on FastMCP)
          #         raise MCPAuthenticationError("Invalid or missing API Key") # Hypothetical error
          #
          #     # Key is valid, proceed with the request
          #     response = await call_next(request)
          #     return response
          #
          # # Register middleware (hypothetical)
          # mcp.add_middleware(api_key_middleware)

          # --- Option 2: Decorator (More likely for tool-level auth) ---
          from functools import wraps

          class MCPAuthenticationError(Exception): # Define a custom error
               pass

          def require_api_key(func):
              @wraps(func)
              async def wrapper(*args, **kwargs):
                  # How to access request context/headers here depends HEAVILY on FastMCP.
                  # Let's assume context is passed implicitly or accessible globally (less ideal)
                  # OR, more realistically, the MCP framework injects context/request info.

                  # ---- Hypothetical context access ---
                  # request_context = FastMCP.get_current_request_context() # Made up!
                  # api_key = request_context.headers.get("X-API-Key")
                  # ---- End Hypothetical ----

                  # --- Simplified/Less Realistic: Pass key as an argument ---
                  # This isn't standard for auth headers, but shows the check logic.
                  # We'd need to modify the tool signature slightly if using this.
                  # api_key = kwargs.get("api_key") # Assume it's passed like any other arg

                  # Let's assume a more plausible scenario where the MCP call handler
                  # somehow makes the key available before calling the decorated function.
                  # Maybe via a special argument injected by the framework?
                  # Assume 'mcp_context' is injected by the framework:
                  mcp_context = kwargs.get('mcp_context', {}) # Hypothetical context object
                  api_key = mcp_context.get("api_key") # Key extracted by framework/middleware

                  if not api_key or api_key not in VALID_API_KEYS:
                       raise MCPAuthenticationError("Invalid or missing API Key")

                  # Key is valid, call the original tool function
                  # Need to remove the context if it wasn't part of original signature
                  original_kwargs = {k: v for k, v in kwargs.items() if k != 'mcp_context'}
                  return await func(*args, **original_kwargs)
              return wrapper

          # Apply the decorator to tools that need protection
          @mcp.tool("protected_tool")
          @require_api_key # Apply decorator
          async def my_protected_tool(some_arg: str) -> str:
               # This code only runs if the API key is valid
               return f"Accessed protected tool with arg: {some_arg}"

          # Tool that doesn't require auth
          @mcp.tool("public_tool")
          async def my_public_tool() -> str:
              return "This is a public tool."

          # Need FastMCP error handling to catch MCPAuthenticationError and return 401/403
          # @mcp.exception_handler(MCPAuthenticationError)
          # async def handle_auth_error(request, exc):
          #    return MCPResponse(status_code=401, content={"error": str(exc)}) # Hypothetical response
          ```
    * **Key Libraries:** `FastMCP`, `os`, `functools` (for decorator).
    * **Data Handling:** Reads API key (e.g., from request header `X-API-Key`). No specific data transformation, focuses
      on control flow.
    * **Error Handling:** Defines and raises a specific `MCPAuthenticationError`. Relies on the framework (`FastMCP`) to
      catch this error and translate it into an appropriate protocol response (e.g., HTTP 401/403).
    * **Security:** The core focus. Securely store the `VALID_API_KEYS` (avoid hardcoding in source). Use a robust
      method for key storage in production (secrets manager, database). Determine the standard way to pass the API key (
      header is common). How middleware/decorators access request data is critical and depends on `FastMCP`.

---

### Production-Ready MCP Server (Lifespan Mgmt)

* **Objective:** Implement robust startup and shutdown procedures for managing resources like database pools, background
  tasks, or model loading.
* **Key Concepts:** Lifespan events (`startup`, `shutdown`), resource initialization and cleanup.
* **Core Implementation Details:**
    * **Framework:** `FastMCP` (*Assumption: FastMCP supports lifespan events similar to ASGI frameworks*).
    * **File Structure:** Single file or separate modules for resource management.
    * **Main Components:**
        * `mcp = FastMCP()`
        * Define startup and shutdown handler functions.
        * Register these handlers with the `FastMCP` application.
          ```python
          import asyncpg
          import os
          import asyncio
          # Assume other imports like httpx clients, ML models etc.

          # Global variables for managed resources
          DB_POOL: Optional[asyncpg.Pool] = None
          HTTP_CLIENT: Optional[httpx.AsyncClient] = None
          BACKGROUND_TASKS = set()

          async def app_startup():
              """Handles application startup logic."""
              print("MCP Server starting up...")
              global DB_POOL, HTTP_CLIENT

              # Initialize Database Pool
              try:
                  DB_POOL = await asyncpg.create_pool(
                      # ... connection details from env vars ...
                      min_size=1, max_size=10 # Example pool size config
                  )
                  print("Database pool initialized.")
              except Exception as e:
                  print(f"ERROR: Failed to initialize database pool: {e}")
                  # Depending on severity, might want to prevent server start
                  # raise RuntimeError("DB Pool init failed") from e

              # Initialize shared HTTP Client (good for reusing connections)
              HTTP_CLIENT = httpx.AsyncClient(timeout=10.0)
              print("Shared HTTP client initialized.")

              # Load ML Models (if applicable)
              # print("Loading ML model...")
              # await load_my_model() # Example async model load
              # print("ML model loaded.")

              # Start background tasks (example: periodic cleanup)
              # task = asyncio.create_task(periodic_cleanup_task())
              # BACKGROUND_TASKS.add(task)
              # task.add_done_callback(BACKGROUND_TASKS.discard) # Auto-remove when done
              # print("Background tasks started.")

              print("MCP Server startup complete.")


          async def app_shutdown():
              """Handles application shutdown logic."""
              print("MCP Server shutting down...")
              global DB_POOL, HTTP_CLIENT

              # Gracefully shutdown background tasks
              # if BACKGROUND_TASKS:
              #    print(f"Cancelling {len(BACKGROUND_TASKS)} background tasks...")
              #    for task in list(BACKGROUND_TASKS): # Iterate over a copy
              #        task.cancel()
              #    await asyncio.gather(*BACKGROUND_TASKS, return_exceptions=True)
              #    print("Background tasks cancelled.")

              # Close shared HTTP Client
              if HTTP_CLIENT:
                  await HTTP_CLIENT.aclose()
                  print("Shared HTTP client closed.")

              # Close Database Pool
              if DB_POOL:
                  await DB_POOL.close()
                  print("Database pool closed.")

              # Unload ML models (if needed)
              # print("Unloading ML model...")
              # await unload_my_model()
              # print("ML model unloaded.")

              print("MCP Server shutdown complete.")

          # Register lifespan handlers with FastMCP (syntax is hypothetical)
          # Option 1: Decorators
          # @mcp.on_event("startup")
          # async def startup_handler():
          #     await app_startup()
          #
          # @mcp.on_event("shutdown")
          # async def shutdown_handler():
          #     await app_shutdown()

          # Option 2: Direct registration
          # mcp.add_event_handler("startup", app_startup)
          # mcp.add_event_handler("shutdown", app_shutdown)

          # Define tools that use these resources
          @mcp.tool("query_db_prod")
          async def query_db_prod_tool(query: str) -> list:
               if not DB_POOL: raise RuntimeError("DB Pool not initialized")
               async with DB_POOL.acquire() as conn:
                   # ... execute query ...
                   return [{"result": "dummy data"}] # Placeholder

          @mcp.tool("call_external_api_prod")
          async def call_external_api_prod_tool(url: str) -> dict:
               if not HTTP_CLIENT: raise RuntimeError("HTTP Client not initialized")
               response = await HTTP_CLIENT.get(url)
               response.raise_for_status()
               return response.json()
          ```
    * **Key Libraries:** `FastMCP`, `asyncio`, `asyncpg`, `httpx`, `os`.
    * **Data Handling:** Primarily concerned with managing the lifecycle of shared resources (pools, clients, models),
      not transforming user data.
    * **Error Handling:** Startup handlers should implement robust error handling (e.g., logging failures, potentially
      preventing server start if critical resources fail). Shutdown handlers should ensure cleanup happens even if
      errors occurred during runtime.
    * **Considerations:** Ensures resources are correctly initialized before requests are handled and properly released
      on shutdown, preventing leaks or dangling connections. Essential for stable, long-running server processes. The
      exact mechanism (`@mcp.on_event`, `mcp.add_event_handler`) depends on `FastMCP`'s design.

---

### HTTP-SSE Deployment & Streaming Resource

* **Objective:** Deploy the MCP server over HTTP and provide a resource that streams data using Server-Sent Events (
  SSE).
* **Key Concepts:** HTTP transport, Server-Sent Events protocol, async generators, yielding chunked responses.
* **Core Implementation Details:**
    * **Framework:** `FastMCP` (*Assumption: FastMCP integrates with an ASGI server like Uvicorn/Hypercorn and provides
      helpers for streaming responses*).
    * **File Structure:** Single file.
    * **Main Components:**
        * `mcp = FastMCP()`
        * An async generator function that yields data chunks.
        * A resource endpoint that returns a streaming response.
        * Running the server via an ASGI server (e.g., `uvicorn`).
          ```python
          import asyncio
          import json
          import datetime
          # Hypothetical StreamingResponse class from FastMCP or an ASGI framework
          # from starlette.responses import StreamingResponse # Example from Starlette
          # Let's assume FastMCP provides something similar:
          from fastmcp.responses import StreamingResponse # HYPOTHETICAL

          async def count_generator(limit: int = 10, delay: float = 0.5):
              """Async generator yielding formatted SSE messages."""
              count = 0
              try:
                  while count < limit:
                      count += 1
                      timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
                      # SSE format: data: <json string>\n\n
                      data_payload = json.dumps({"count": count, "timestamp": timestamp})
                      sse_message = f"data: {data_payload}\n\n"
                      yield sse_message
                      await asyncio.sleep(delay)
                  # Optionally send a closing event
                  yield "event: close\ndata: Stream finished\n\n"
              except asyncio.CancelledError:
                  print("Streaming generator cancelled.")
                  yield "event: error\ndata: Stream cancelled by server\n\n"
                  raise # Re-raise cancellation


          # Define a resource that returns a streaming response
          # The path/method depends on how FastMCP handles resources vs tools for streaming
          # Let's assume a GET request to a resource path triggers the stream.
          @mcp.resource("/stream/counter") # Or perhaps @mcp.get("/stream/counter")
          class CounterStreamResource:
               # Method name might vary (e.g., stream, get, handle)
              async def get(self) -> StreamingResponse: # Return type hints the streaming nature
                   """Initiates an SSE stream of counter events."""
                   # The framework needs to know how to handle an async generator
                   # by wrapping it in a streaming response object.
                   return StreamingResponse(
                       count_generator(limit=20, delay=1.0),
                       media_type="text/event-stream"
                   )

          # --- Running the Server ---
          # This part usually involves an ASGI server like uvicorn
          # In your main execution block:
          # if __name__ == "__main__":
          #    import uvicorn
          #    # Assuming 'mcp' is the FastMCP ASGI application instance
          #    uvicorn.run(mcp, host="0.0.0.0", port=8000, lifespan="on") # lifespan="on" enables startup/shutdown

          ```
    * **Key Libraries:** `FastMCP`, `asyncio`, `json`, `datetime`. An ASGI server (`uvicorn`, `hypercorn`) is needed for
      deployment. Potentially ASGI framework components (`starlette.responses` if FastMCP builds on it).
    * **Data Handling:** Generates data within an `async def` function using `yield`. Formats data according to the SSE
      specification (`data: ...\n\n`). The framework handles sending these chunks over HTTP.
    * **Error Handling:** The generator should handle `asyncio.CancelledError` if the client disconnects or the server
      shuts down during streaming. The framework handles underlying network errors.
    * **Deployment:** Requires running via an ASGI server (like `uvicorn your_server_file:mcp --reload`) instead of
      `mcp.run_stdio()`. The server listens on an HTTP port.
    * **Client Interaction:** Clients connect via HTTP GET to `/stream/counter` and need to handle the
      `text/event-stream` content type, parsing messages as they arrive.

