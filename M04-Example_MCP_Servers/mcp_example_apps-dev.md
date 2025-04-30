# MCP Example Apps

| #   | Project Title (easiest → hardest)               | Real-world Scenario & New Skill                                                                                                            | Core Tasks & Code Highlights                                                                                                                                                                                                                                                                       |
| --- | ----------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | **Echo + Calculator MCP Server**                | _Kick-off_: show the protocol with two toy tools you can unit-test instantly.                                                              | _ `FastMCP("demo_server")` <br> _ `@mcp.tool() def echo(msg:str)->str` <br> _ `@mcp.tool() def add(a:int,b:int)->int` <br> _ Run via `mcp.run(transport="stdio")` and call with a 4-line Python client.                                                                                            |
| 2   | **Weather-Forecast Tool**                       | Pull live weather from api.weather.gov – first **async HTTP** call & JSON parsing.                                                         | _ Add `httpx.AsyncClient()` inside an `async def get_forecast(lat,lon)`. <br> _ Format multi-period forecast into human-readable string. <br> \* Handle 404s ⇒ return “No forecast”.                                                                                                               |
| 3   | **Secure File-Reader**                          | An AI assistant needs to read docs _only inside_ a project folder. Teaches **roots / path-sanitisation**.                                  | _ `ALLOWED_ROOT = Path(...).resolve()` <br> _ Reject paths outside root. <br> _ Truncate files >5 kB. <br> _ Optional `find_files(keyword)` search.                                                                                                                                                |
| 4   | **CSV Quick-Insights Tool**                     | Data analyst asks: “give me stats on sales.csv”. Learn **pandas** + on-the-fly summarisation.                                              | _ `@mcp.tool() def csv_summary(path:str)->str` <br> _ Read with `pd.read_csv` (after root check). <br> \* Return rows, cols, `head()`, `dtypes`, and simple numeric `describe()`.                                                                                                                  |
| 5   | **PostgreSQL Query Server**                     | Backend engineer wants AI to run safe read-only SQL. Introduces **database drivers + parameter whitelists**.                               | _ Pool via `asyncpg.create_pool()`. <br> _ `Workspace_rows(query:str)` – allow only `SELECT …` (regex guard). <br> \* Return rows as `list[dict]` (JSON-serialisable).                                                                                                                             |
| 6   | **Stock-Price + Sentiment Bundler**             | Combines two APIs (Yahoo Finance & NewsAPI) → one MCP tool that fuses data. Shows **multi-call aggregation**.                              | _ `get_stock_insights(ticker:str)` calls `yfinance` & `newsapi`. <br> _ Run simple TextBlob polarity over headlines; return dict `{price, headlines, sentiment_score}`.                                                                                                                            |
| 7   | **GitHub PR Fetcher**                           | Dev-tools use case. Adds **auth via env tokens** & paginated API calls. (Foundation for larger case study.)                                | _ `GITHUB_TOKEN` from `.env`. <br> _ `Workspace_pr(owner,repo,number)` collects title/body + first N file diffs (truncate patches). <br> \* Return structured dict.                                                                                                                                |
| 8   | **Notion Page Creator**                         | Demonstrate a **write-back** workflow. AI can persist results.                                                                             | _ `NOTION_API_KEY`, `NOTION_PAGE_ID` env vars. <br> _ `create_notion_page(title,content)` via `notion_client.pages.create`. <br> \* Return success / error string.                                                                                                                                 |
| 9   | **PR Assistant End-to-End (Client + Server)**   | Glue Levels 7 & 8. Build a tiny CLI chat loop where Claude (or GPT-4o) calls both tools to summarise a PR then log to Notion.              | **Client side:** <br> _ Spawn MCP server subprocess. <br> _ `anthropic` API call with tool schema in system prompt. <br> _ Regex/JSON parse for `{"tool": …}` pattern → `session.run_tool()`. <br> _ Loop until LLM replies without tool.                                                          |
| 10  | **Local Llama 2 Agent with Ollama + LangChain** | Privacy-first viewers run everything offline. Show **Ollama LLM + LangChain MCP adapter**.                                                 | _ `llm = Ollama(model="llama2")` <br> _ `tools = load_tools_from_mcp(server_path)` <br> _ `agent = initialize_agent(tools,llm,verbose=True)` <br> _ Ask math or weather questions, watch model invoke tools.                                                                                       |
| 11  | **Multi-Server, Multi-Step Research Agent**     | Orchestrate two separate servers: (a) _Brave Search_ (npm package) and (b) our Notion server. Agent finds latest AI article → saves TL;DR. | _ Start Brave MCP via `npx -y @mcp/brave-search`. <br> _ Use LangChain agent to chain `search_web` → summarise → `create_notion_page`. <br> \* Highlight verbose reasoning logs.                                                                                                                   |
| 12  | **HTTP-SSE Deployment & Streaming Resource**    | Production-grade pattern. Serve a **large PDF** as a streamed resource over HTTP and connect from a remote client.                         | **Server:** <br> _ `mcp.run(transport="sse",host="0.0.0.0",port=5000)` <br> _ `@mcp.resource("annual_report")` returns file handle. <br> **Client:** <br> \* `session.connect(HttpServerParameters("https://mydomain.com:5000"))` then `read_resource("annual_report",chunk_size=2048)` in a loop. |

## ## **. Basic Calculator MCP Server**

- **Goal:** Create the simplest possible functional MCP server.
- **Concept:** Expose a basic arithmetic function as an MCP Tool.
- **Core Task:** Use `FastMCP` from the `mcp` Python SDK to create a server instance. Define a Python function (e.g., `add(a: int, b: int) -> int`) and expose it using the `@mcp.tool()` decorator.[1, 2] Run the server locally via stdio.
- **Difficulty:** Easy

## **. User Profile MCP Server (Tool + Resource)**

- **Goal:** Demonstrate exposing both actions (Tools) and data (Resources).
- **Concept:** Add the ability to retrieve simple user data alongside the calculator function.
- **Core Task:** Extend the Calculator server from Example 1. Add a new function (e.g., `get_user_profile(user_id: str) -> str`) that returns mock profile data. Expose this function using the `@mcp.resource("users://{user_id}/profile")` decorator, showcasing dynamic path parameters.[2]
- **Difficulty:** Easy

## **. Simple MCP Client (Connect & Interact)**

- **Goal:** Show how to connect to an MCP server and use its capabilities.
- **Concept:** Build a basic client application to interact with the server created in Example 2.
- **Core Task:** Write a Python script using `ClientSession` and `stdio_client` from the `mcp` SDK.[2] Connect to the running User Profile server (via stdio). Initialize the session, list the available tools and resources, call the `add` tool with sample numbers, and read the `users://some_user/profile` resource.[2] Print the results.
- **Difficulty:** Easy

## **. Code Review Helper MCP Server (Tool + Resource + Prompt)**

- **Goal:** Introduce MCP Prompts for guiding interactions.
- **Concept:** Create a server that offers a tool to analyze code complexity (mock analysis) and a prompt to guide the user on how to request a review.
- **Core Task:** Build a new `FastMCP` server. Include a mock tool `@mcp.tool() def analyze_complexity(code: str) -> str: return "Complexity: Medium"`. Add a prompt using `@mcp.prompt() def review_code(code: str) -> str: return f"Please review this code for style and complexity:\n\n{code}"`.[2] Include a simple resource for configuration if desired.
- **Difficulty:** Easy-Medium

## **. Chatbot Context Server & Client**

- **Goal:** Demonstrate a practical use case for MCP Resources in enhancing conversational AI.
- **Concept:** An MCP server provides dynamic context (like user preferences) that a separate chatbot client fetches before generating a response.
- **Core Task:**
  - **Server:** Create a `FastMCP` server exposing a resource like `@mcp.resource("/context/user/{user_id}/preferences") def get_prefs(user_id: str) -> dict: return {"topic": "technology", "tone": "formal"}`.
  - **Client:** Write a simple client script that takes a user ID, connects to the server, calls `read_resource` to get preferences [2], and then constructs a prompt for a (simulated or placeholder) LLM incorporating those preferences (e.g., "User prefers technology topics and a formal tone. Answer their question:..."). This focuses on the MCP context retrieval part.[3, 4]
- **Difficulty:** Medium

## **. Ollama Client with MCP Context**

- **Goal:** Show how MCP can provide external context to locally running LLMs.
- **Concept:** Fetch data using an MCP client and inject it into a prompt sent to an Ollama model via its API.
- **Core Task:** Write a Python script that:
  1. Connects to an existing MCP server (e.g., the User Profile server from Example 2) using `ClientSession`.[2]
  2. Calls `read_resource` to fetch data (e.g., user profile).
  3. Formats the fetched data into a string.
  4. Uses the `requests` library or an Ollama client library to make a POST request to the Ollama API (`/api/generate` or `/api/chat`).[5, 6]
  5. Includes the fetched MCP data within the `prompt` or `messages` payload sent to Ollama.[5, 7, 8] Print the Ollama response.
- **Difficulty:** Medium

## **. Langchain Agent with MCP Calculator Tool**

- **Goal:** Integrate a basic MCP tool into a standard Langchain agent workflow.
- **Concept:** Use the `langchain-mcp-adapters` library to make an MCP tool available to a Langchain agent.
- **Core Task:**
  1. Ensure the Calculator MCP server (Example 1) is running.
  2. Install `langchain-mcp-adapters`, `langchain-openai` (or another LLM provider).[1, 9]
  3. Use `MultiServerMCPClient` from the adapter library to connect to the calculator server via stdio.[1, 10]
  4. Load the MCP tools into Langchain `BaseTool` objects using the client.[11, 1]
  5. Create a simple Langchain ReAct agent (e.g., using `create_react_agent`) passing the LLM and the loaded MCP tool(s).[1]
  6. Invoke the agent with a prompt like "What is 15 plus 27?" and observe it using the MCP `add` tool.[1, 12]
- **Difficulty:** Medium

## **. RAG Enhancement with MCP Real-time Data**

- **Goal:** Improve a RAG pipeline by fetching dynamic data via MCP.
- **Concept:** Supplement vector store retrieval with fresh data from an MCP resource.
- **Core Task:**
  - **Server:** Create an MCP server exposing a resource for mock real-time data, e.g., `@mcp.resource("/stocks/{ticker}") def get_stock_price(ticker: str) -> str: return f"Price for {ticker}: ${random.uniform(100, 500):.2f}"`.
  - **Client (RAG):** Set up a basic Langchain RAG pipeline (e.g., using a simple vector store). Modify the pipeline (or use an agent step before generation) to:
    1. Identify if the query mentions a stock ticker.
    2. If yes, connect to the MCP stock server using `ClientSession`.[2]
    3. Call `read_resource` to get the mock price.
    4. Add this fetched price information to the context passed to the LLM along with documents retrieved from the vector store.[13, 3]
- **Difficulty:** Medium-Hard

## **. Simple Langgraph Workflow Calling an MCP Tool**

- **Goal:** Introduce Langgraph by showing an MCP tool execution within a graph node.
- **Concept:** Build a minimal state machine where one state transition involves calling an MCP tool.
- **Core Task:**
  1. Ensure the Calculator MCP server (Example 1) is running.
  2. Use `langchain-mcp-adapters` and `MultiServerMCPClient` to load the `add` tool (as in Example 7).[1, 10]
  3. Define a simple Langgraph `StateGraph` with a state schema.[14, 15]
  4. Create a node function that takes the state, calls the MCP `add` tool using the loaded Langchain tool object, and updates the state with the result.
  5. Define edges and compile the graph.[14]
  6. Invoke the graph with initial inputs and observe the MCP tool being called within the node.
- **Difficulty:** Medium-Hard

## **. Secure MCP Server (API Key Auth)**

- **Goal:** Demonstrate a basic security pattern for MCP servers.
- **Concept:** Add a simple check for an API key before allowing a tool to execute. (Note: MCP spec doesn't enforce auth; this is a server-side implementation pattern).
- **Core Task:** Modify an existing server (e.g., Calculator Server from Example 1).
  - Add an `api_key: str` parameter to the `add` tool function signature.
  - Inside the function, add a simple check: `if api_key!= "SECRET_KEY": raise PermissionError("Invalid API Key")`.
  - Modify the client (Example 3) to pass the correct API key in the `arguments` dictionary when calling `call_tool`.[2] Show that calls fail without the correct key.
- **Difficulty:** Hard (Conceptual implementation, as SDK/protocol doesn't dictate _how_ auth is done [16, 2])

## **. Production-Ready MCP Server (Lifespan Management)**

- **Goal:** Implement server startup/shutdown logic for resource management.
- **Concept:** Use `FastMCP`'s lifespan context manager to handle resources like database connections.
- **Core Task:** Create an MCP server.
  - Define `async def startup():` and `async def shutdown():` functions. Print messages indicating startup/shutdown. (Simulate acquiring/releasing a resource like a DB connection pool).
  - Use the `mcp.server.fastmcp.lifespan` async context manager within the main server run function to wrap the server execution, passing the startup/shutdown functions.[2]
  - Run the server and observe the startup/shutdown messages.
- **Difficulty:** Hard

## **. Langgraph with External MCP State Management**

- **Goal:** Use MCP as an interface for managing state externally from a Langgraph workflow.
- **Concept:** Create a dedicated MCP server for state operations (get/set) and have a Langgraph agent interact with it.
- **Core Task:**
  - **MCP State Server:** Build a `FastMCP` server with:
    - `@mcp.resource("/state/{session_id}") async def get_state(session_id: str) -> dict:` (reads from a simple in-memory dict).
    - `@mcp.tool() async def set_state(session_id: str, data: dict) -> bool:` (writes to the dict).
  - **Langgraph Agent:**
    1. Use `langchain-mcp-adapters` to connect to the state server and load the `get_state` (as a resource-reading tool if possible, or adapt) and `set_state` tools.[1]
    2. Define a Langgraph `StateGraph`.[17, 14]
    3. Create nodes that use the loaded MCP tools to read the state from the server at the beginning and write updated state back to the server at the end of their execution.[18, 19, 20]
- **Difficulty:** Hard

## **. Multi-Tool Langgraph Research Agent**

- **Goal:** Build a more complex agent that orchestrates multiple different MCP tools.
- **Concept:** A Langgraph agent uses tools from potentially different MCP servers (e.g., web search, file reading) to complete a research task.
- **Core Task:**
  1. **Servers:** Create two simple, separate MCP servers:
     - `Search Server`: `@mcp.tool() def web_search(query: str) -> str: return f"Mock search results for {query}"`.
     - `File Server`: `@mcp.tool() def read_file(path: str) -> str: return f"Mock content of file {path}"`. Run both servers (e.g., on different ports if using SSE, or manage separate stdio processes).
  2. **Langgraph Agent:**
     - Use `MultiServerMCPClient` to connect to _both_ the Search Server and File Server.[1, 10]
     - Load all tools from both servers using the adapter.
     - Design a Langgraph `StateGraph` with nodes for planning (deciding which tool to use), tool execution, and result synthesis.[17, 14]
     - Implement conditional edges based on the plan or intermediate results (e.g., if plan says "search", go to search node).
     - Invoke the agent with a research query (e.g., "Search for MCP and read the intro file") and observe it calling the appropriate tools on the correct servers.[9, 17]
- **Difficulty:** Very Hard
  Okay, let's outline some concrete coding examples based on our previous discussion. These are designed as practical projects you could feature in your YouTube videos, progressing from fundamental MCP concepts to more complex integrations with Langchain, Ollama, and Langgraph.

Here are 13 real-world coding examples, ordered from easiest to hardest:

---

## **. Basic Calculator MCP Server**

- **Goal:** Create the simplest possible functional MCP server.
- **Concept:** Expose a basic arithmetic function as an MCP Tool.
- **Core Task:** Use `FastMCP` from the `mcp` Python SDK to create a server instance. Define a Python function (e.g., `add(a: int, b: int) -> int`) and expose it using the `@mcp.tool()` decorator.[1, 2] Run the server locally via stdio.
- **Difficulty:** Easy

## **. User Profile MCP Server (Tool + Resource)**

- **Goal:** Demonstrate exposing both actions (Tools) and data (Resources).
- **Concept:** Add the ability to retrieve simple user data alongside the calculator function.
- **Core Task:** Extend the Calculator server from Example 1. Add a new function (e.g., `get_user_profile(user_id: str) -> str`) that returns mock profile data. Expose this function using the `@mcp.resource("users://{user_id}/profile")` decorator, showcasing dynamic path parameters.[2]
- **Difficulty:** Easy

## **. Simple MCP Client (Connect & Interact)**

- **Goal:** Show how to connect to an MCP server and use its capabilities.
- **Concept:** Build a basic client application to interact with the server created in Example 2.
- **Core Task:** Write a Python script using `ClientSession` and `stdio_client` from the `mcp` SDK.[2] Connect to the running User Profile server (via stdio). Initialize the session, list the available tools and resources, call the `add` tool with sample numbers, and read the `users://some_user/profile` resource.[2] Print the results.
- **Difficulty:** Easy

## **. Code Review Helper MCP Server (Tool + Resource + Prompt)**

- **Goal:** Introduce MCP Prompts for guiding interactions.
- **Concept:** Create a server that offers a tool to analyze code complexity (mock analysis) and a prompt to guide the user on how to request a review.
- **Core Task:** Build a new `FastMCP` server. Include a mock tool `@mcp.tool() def analyze_complexity(code: str) -> str: return "Complexity: Medium"`. Add a prompt using `@mcp.prompt() def review_code(code: str) -> str: return f"Please review this code for style and complexity:\n\n{code}"`.[2] Include a simple resource for configuration if desired.
- **Difficulty:** Easy-Medium

## **. Chatbot Context Server & Client**

- **Goal:** Demonstrate a practical use case for MCP Resources in enhancing conversational AI.
- **Concept:** An MCP server provides dynamic context (like user preferences) that a separate chatbot client fetches before generating a response.
- **Core Task:**
  - **Server:** Create a `FastMCP` server exposing a resource like `@mcp.resource("/context/user/{user_id}/preferences") def get_prefs(user_id: str) -> dict: return {"topic": "technology", "tone": "formal"}`.
  - **Client:** Write a simple client script that takes a user ID, connects to the server, calls `read_resource` to get preferences [2], and then constructs a prompt for a (simulated or placeholder) LLM incorporating those preferences (e.g., "User prefers technology topics and a formal tone. Answer their question:..."). This focuses on the MCP context retrieval part.[3, 4]
- **Difficulty:** Medium

## **. Ollama Client with MCP Context**

- **Goal:** Show how MCP can provide external context to locally running LLMs.
- **Concept:** Fetch data using an MCP client and inject it into a prompt sent to an Ollama model via its API.
- **Core Task:** Write a Python script that:
  1. Connects to an existing MCP server (e.g., the User Profile server from Example 2) using `ClientSession`.[2]
  2. Calls `read_resource` to fetch data (e.g., user profile).
  3. Formats the fetched data into a string.
  4. Uses the `requests` library or an Ollama client library to make a POST request to the Ollama API (`/api/generate` or `/api/chat`).[5, 6]
  5. Includes the fetched MCP data within the `prompt` or `messages` payload sent to Ollama.[5, 7, 8] Print the Ollama response.
- **Difficulty:** Medium

## **. Langchain Agent with MCP Calculator Tool**

- **Goal:** Integrate a basic MCP tool into a standard Langchain agent workflow.
- **Concept:** Use the `langchain-mcp-adapters` library to make an MCP tool available to a Langchain agent.
- **Core Task:**
  1. Ensure the Calculator MCP server (Example 1) is running.
  2. Install `langchain-mcp-adapters`, `langchain-openai` (or another LLM provider).[1, 9]
  3. Use `MultiServerMCPClient` from the adapter library to connect to the calculator server via stdio.[1, 10]
  4. Load the MCP tools into Langchain `BaseTool` objects using the client.[11, 1]
  5. Create a simple Langchain ReAct agent (e.g., using `create_react_agent`) passing the LLM and the loaded MCP tool(s).[1]
  6. Invoke the agent with a prompt like "What is 15 plus 27?" and observe it using the MCP `add` tool.[1, 12]
- **Difficulty:** Medium

## **. RAG Enhancement with MCP Real-time Data**

- **Goal:** Improve a RAG pipeline by fetching dynamic data via MCP.
- **Concept:** Supplement vector store retrieval with fresh data from an MCP resource.
- **Core Task:**
  - **Server:** Create an MCP server exposing a resource for mock real-time data, e.g., `@mcp.resource("/stocks/{ticker}") def get_stock_price(ticker: str) -> str: return f"Price for {ticker}: ${random.uniform(100, 500):.2f}"`.
  - **Client (RAG):** Set up a basic Langchain RAG pipeline (e.g., using a simple vector store). Modify the pipeline (or use an agent step before generation) to:
    1. Identify if the query mentions a stock ticker.
    2. If yes, connect to the MCP stock server using `ClientSession`.[2]
    3. Call `read_resource` to get the mock price.
    4. Add this fetched price information to the context passed to the LLM along with documents retrieved from the vector store.[13, 3]
- **Difficulty:** Medium-Hard

## **. Simple Langgraph Workflow Calling an MCP Tool**

- **Goal:** Introduce Langgraph by showing an MCP tool execution within a graph node.
- **Concept:** Build a minimal state machine where one state transition involves calling an MCP tool.
- **Core Task:**
  1. Ensure the Calculator MCP server (Example 1) is running.
  2. Use `langchain-mcp-adapters` and `MultiServerMCPClient` to load the `add` tool (as in Example 7).[1, 10]
  3. Define a simple Langgraph `StateGraph` with a state schema.[14, 15]
  4. Create a node function that takes the state, calls the MCP `add` tool using the loaded Langchain tool object, and updates the state with the result.
  5. Define edges and compile the graph.[14]
  6. Invoke the graph with initial inputs and observe the MCP tool being called within the node.
- **Difficulty:** Medium-Hard

## **. Secure MCP Server (API Key Auth)**

- **Goal:** Demonstrate a basic security pattern for MCP servers.
- **Concept:** Add a simple check for an API key before allowing a tool to execute. (Note: MCP spec doesn't enforce auth; this is a server-side implementation pattern).
- **Core Task:** Modify an existing server (e.g., Calculator Server from Example 1).
  - Add an `api_key: str` parameter to the `add` tool function signature.
  - Inside the function, add a simple check: `if api_key!= "SECRET_KEY": raise PermissionError("Invalid API Key")`.
  - Modify the client (Example 3) to pass the correct API key in the `arguments` dictionary when calling `call_tool`.[2] Show that calls fail without the correct key.
- **Difficulty:** Hard (Conceptual implementation, as SDK/protocol doesn't dictate _how_ auth is done [16, 2])

## **. Production-Ready MCP Server (Lifespan Management)**

- **Goal:** Implement server startup/shutdown logic for resource management.
- **Concept:** Use `FastMCP`'s lifespan context manager to handle resources like database connections.
- **Core Task:** Create an MCP server.
  - Define `async def startup():` and `async def shutdown():` functions. Print messages indicating startup/shutdown. (Simulate acquiring/releasing a resource like a DB connection pool).
  - Use the `mcp.server.fastmcp.lifespan` async context manager within the main server run function to wrap the server execution, passing the startup/shutdown functions.[2]
  - Run the server and observe the startup/shutdown messages.
- **Difficulty:** Hard

## **. Langgraph with External MCP State Management**

- **Goal:** Use MCP as an interface for managing state externally from a Langgraph workflow.
- **Concept:** Create a dedicated MCP server for state operations (get/set) and have a Langgraph agent interact with it.
- **Core Task:**
  - **MCP State Server:** Build a `FastMCP` server with:
    - `@mcp.resource("/state/{session_id}") async def get_state(session_id: str) -> dict:` (reads from a simple in-memory dict).
    - `@mcp.tool() async def set_state(session_id: str, data: dict) -> bool:` (writes to the dict).
  - **Langgraph Agent:**
    1. Use `langchain-mcp-adapters` to connect to the state server and load the `get_state` (as a resource-reading tool if possible, or adapt) and `set_state` tools.[1]
    2. Define a Langgraph `StateGraph`.[17, 14]
    3. Create nodes that use the loaded MCP tools to read the state from the server at the beginning and write updated state back to the server at the end of their execution.[18, 19, 20]
- **Difficulty:** Hard

## **. Multi-Tool Langgraph Research Agent**

- **Goal:** Build a more complex agent that orchestrates multiple different MCP tools.
- **Concept:** A Langgraph agent uses tools from potentially different MCP servers (e.g., web search, file reading) to complete a research task.
- **Core Task:**
  1. **Servers:** Create two simple, separate MCP servers:
     - `Search Server`: `@mcp.tool() def web_search(query: str) -> str: return f"Mock search results for {query}"`.
     - `File Server`: `@mcp.tool() def read_file(path: str) -> str: return f"Mock content of file {path}"`. Run both servers (e.g., on different ports if using SSE, or manage separate stdio processes).
  2. **Langgraph Agent:**
     - Use `MultiServerMCPClient` to connect to _both_ the Search Server and File Server.[1, 10]
     - Load all tools from both servers using the adapter.
     - Design a Langgraph `StateGraph` with nodes for planning (deciding which tool to use), tool execution, and result synthesis.[17, 14]
     - Implement conditional edges based on the plan or intermediate results (e.g., if plan says "search", go to search node).
     - Invoke the agent with a research query (e.g., "Search for MCP and read the intro file") and observe it calling the appropriate tools on the correct servers.[9, 17]
- **Difficulty:** Very Hard
