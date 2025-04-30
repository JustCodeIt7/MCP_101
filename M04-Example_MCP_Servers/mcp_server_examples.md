| Rank | Project / Example | Key new concepts that make it harder than the one above |
|------| ---------------- | -------------------------------------------------------- |
| 1    | **Basic / Echo + Calculator MCP Server** | First look at `FastMCP`, single-file, two toy tools, runs over stdio. |
| 2    | **User Profile MCP Server** | Adds a dynamic **Resource** path and a second domain object. |
| 3    | **Simple MCP Client** | Shows `ClientSession` & `stdio_client`, but no server logic. |
| 4    | **Code Review Helper MCP Server** | Introduces the `@mcp.prompt` decorator; still “string-in, string-out”. |
| 5    | **Weather-Forecast Tool** | First **async HTTP** call and JSON parsing, plus 404 handling. |
| 6    | **Multi-Tool LangGraph Research Agent** | Combines *several* MCP servers, conditional branches, planning node, and result synthesis in LangGraph—hardest because it mixes everything above at once. |
| 7    | **Secure File-Reader** | Path-sanitisation (root locking) and file-size truncation. |
| 22   | **Multi-Server, Multi-Step Research Agent** | Coordinates **two** running MCP servers inside one LangChain agent. |
| 8    | **CSV Quick-Insights Tool** | Brings in **pandas** for I/O and summarisation. |
| 15   | **LangChain Agent with MCP Calculator Tool** | Bridges MCP → LangChain; need tool adapters & agent loop. |
| 23   | **HTTP-SSE Deployment & Streaming Resource** | Network transport change, chunked streaming, remote client connect. |
| 8    | **Chatbot Context Server & Client** | Resources feed an external LLM; requires coordinating two processes. |
| 9    | **Ollama Client with MCP Context** | Adds a local LLM REST call and prompt templating. |
| 10   | **Stock-Price + Sentiment Bundler** | Aggregates **two** APIs and does basic NLP polarity scoring. |
| 11   | **PostgreSQL Query Server** | Real DB pool (`asyncpg`) and a read-only SQL guard with regex. |
| 12   | **Notion Page Creator** | First **write-back** workflow; OAuth-style secret management. |
| 13   | **Secure MCP Server (API-Key Auth)** | Adds explicit request-level auth & error handling paths. |
| 14   | **Production-Ready MCP Server (Lifespan Mgmt)** | Requires startup/shutdown hooks and resource cleanup. |
| 16   | **RAG Enhancement with MCP Real-time Data** | Orchestrates vector store retrieval *and* live MCP context. |
| 17   | **Simple LangGraph Workflow Calling an MCP Tool** | First LangGraph state machine; tool call embedded in a node. |
| 19   | **Local Llama 2 Agent (Ollama + LangChain)** | Offline LLM, larger model weights, plus LangChain tool harness. |
| 20   | **PR Assistant End-to-End (Client + Server)** | Spawns the server as a subprocess and mediates LLM ↔ tools loops. |
| 21   | **LangGraph with External MCP State Management** | Two-way state read/write over MCP inside a multi-node graph. |


Below is a **project-by-project implementation roadmap** that you can follow from top to bottom in the exact order you supplied.  
For each entry you’ll find:

1. **What you’ll build**  – the finished behaviour.  
2. **Libraries & setup**  – pip installs and .env variables.  
3. **Suggested file layout**  – keeps every repo consistent.  
4. **Step-by-step coding plan**  – in the sequence you’ll actually type.  
5. **Smoke-test commands**  – one liner(s) that prove it works.  
6. **Stretch goals / next steps**  – useful if you want to re-record upgrades later.

---

## 1  Basic / Echo + Calculator MCP Server
**Concept :** hello-world for FastMCP – one file, two synchronous tools, stdio transport.

| Section | Details |
|---------|---------|
| Libs    | `pip install mcp fastmcp` |
| Layout |```\ncalculator_server/\n├─ server.py\n└─ requirements.txt```|
| Plan | 1. `from mcp.server.fastmcp import FastMCP`\n2. Instantiate `FastMCP("demo_server")`.\n3. Decorate two plain functions:\n   ```python\n   @mcp.tool()\n   def echo(msg:str)->str: return msg\n   @mcp.tool()\n   def add(a:int,b:int)->int: return a+b\n   ```\n4. `if __name__ == "__main__": mcp.run(transport="stdio")` |
| Test |```bash\npython server.py | python - <<'EOF'\nfrom mcp import ClientSession, stdio_client\nwith ClientSession(stdio_client.spawn(\"python server.py\")) as s:\n    print(s.call_tool(\"echo\", msg=\"hi\"))\n    print(s.call_tool(\"add\", a=2, b=3))\nEOF```|
| Stretch | Switch to `transport="sse"` and bind to `localhost:5000`. |

---

## 2  User Profile MCP Server
Adds a **resource** endpoint.

| Section | Details |
|---------|---------|
| Libs | same as #1 |
| Layout |```\nuser_profile_server/\n├─ server.py```|
| Plan | 1. Copy code from #1.<br>2. Add mock data dict `USERS`.<br>3. Decorate a resource: `@mcp.resource(\"users://{user_id}/profile\")` returning JSON-serialisable `dict`.<br>4. Keep calculator `add` so client demos both tool + resource. |
| Test | Use `ClientSession.read_resource("users://alice/profile")`. |
| Stretch | Make `users.json` on disk and load at startup. |

---

## 3  Simple MCP Client (stdio)
Pure client—no new server code.

| Section | Details |
| Libraries | `pip install mcp` |
| Layout |```\nprofile_client/\n└─ client.py```|
| Plan | 1. Spawn the server from #2 in a subprocess (or expect user to run it).<br>2. Connect with `stdio_client.spawn(...)`.<br>3. `session.list_tools()`, `call_tool`, `read_resource`.<br>4. Pretty-print results. |
| Test | `python client.py` should show `5` and the profile JSON. |
| Stretch | Add CLI arguments: `--user-id`, `--a`, `--b`. |

---

## 4  Code Review Helper MCP Server
First use of `@mcp.prompt`.

| Section | Details |
| Libs | same as #1 |
| Layout |```\ncode_review_server/\n└─ server.py```|
| Plan | 1. Fresh `FastMCP("code_review")` instance.<br>2. Tool: `analyze_complexity(code:str)` – returns dummy string.<br>3. Prompt: `@mcp.prompt() def review_code(code:str)->str` – formats guidance.<br>4. Optional read-only `config.toml` resource. |
| Test | Use local client to call prompt; confirm returned template. |
| Stretch | Replace dummy complexity with `radon` cyclomatic complexity. |

---

## 5  Weather-Forecast Tool
**First async HTTP request & JSON parse.**

| Section | Details |
| Libs | `pip install mcp httpx` |
| Layout |```\nweather_server/\n└─ server.py```|
| Plan | 1. Use `httpx.AsyncClient()` inside `async def get_forecast(lat:float, lon:float)->str`.<br>2. Endpoint: `https://api.weather.gov/points/{lat},{lon}` → second call to forecast URL.<br>3. Handle non-200 / KeyErrors => `"No forecast available"`.<br>4. Register as `@mcp.tool()` **async** function.<br>5. Run over stdio. |
| Test | `call_tool("get_forecast", lat=39.7, lon=-104.9)` prints first 3 periods. |
| Stretch | Add caching via `functools.lru_cache` for 5 min. |

---

## 6  Multi-Tool LangGraph Research Agent (**hard**)  
Mixes several MCP servers, conditional branching, and result synthesis in LangGraph.

| Section | Details |
| Libs | `pip install langgraph langchain langchain-openai langchain-mcp-adapters` |
| Layout |```\nresearch_agent/\n├─ servers/\n│  ├─ search_server.py\n│  └─ file_server.py\n├─ graph_agent.py\n└─ README.md```|
| Plan | **Servers:**<br>• `search_server.py` → `web_search(query:str)` dummy returns.<br>• `file_server.py` → `read_file(path:str)` dummy returns.<br><br>**Graph:**<br>1. Spawn both servers with `subprocess.Popen`.<br>2. `client = MultiServerMCPClient([...])` loads all tools as LangChain `BaseTool` objects.<br>3. Define `WorkflowState(TypedDict)` with `question`, `plan`, `intermediate`, `answer`.<br>4. Nodes:<br>   * *Planner* (LLM) → decides `"search"` vs `"read"`.<br>   * *ToolExec* → runs chosen tool.<br>   * *Synthesise* → LLM combines pieces.<br>5. Conditional edges based on planner output.<br>6. Compile & invoke.<br><br>**Cleanup:** ensure `finally: proc.terminate()` on servers. |
| Test | Ask: “Search for MCP intro and read README.md” – observe both tools used. |
| Stretch | Replace dummy servers with real Brave Search + local file system read. |

---

## 7  Secure File-Reader MCP Server
Focus on path security.

| Section | Details |
| Libs | `pip install mcp` |
| Layout |```\nfile_reader_server/\n├─ server.py\n└─ docs/  # safe root```|
| Plan | 1. `ALLOWED_ROOT = Path("docs").resolve()`.<br>2. Tool: `read_file(path:str)`.<br>3. Inside: `abs_path = (ALLOWED_ROOT / path).resolve(); assert abs_path.is_relative_to(ALLOWED_ROOT)`.<br>4. Truncate to 5 kB: `content[:5_000]`.<br>5. Optional `find_files(keyword)` that walks tree. |
| Test | Call with `../secret.txt` → get PermissionError. |
| Stretch | Add MIME sniffing & refuse binary files. |

---

## 8  CSV Quick-Insights Tool
Uses pandas.

| Section | Details |
| Libs | `pip install mcp pandas` |
| Layout |```\ncsv_insights_server/\n└─ server.py```|
| Plan | Tool `csv_summary(path:str)->str`:<br>1. Root-check like #7.<br>2. `df = pd.read_csv(path)`.<br>3. Build summary string: rows, cols, `df.head(5).to_markdown()`, `df.dtypes`, `df.describe().to_markdown()`. |
| Test | Place `sales.csv` in root and call tool. |
| Stretch | Return a structured dict; let client pretty-print. |

---

## 15  LangChain Agent with MCP Calculator Tool
Bridges MCP → LangChain.

| Section | Details |
| Libs | `pip install langchain langchain-openai langchain-mcp-adapters` |
| Layout |```\ncalculator_agent/\n└─ agent.py```|
| Plan | 1. Ensure server #1 is running (or spin up inside script).<br>2. `client = MultiServerMCPClient([\"stdio:python server.py\"])`.<br>3. `tools = client.load_tools()`.<br>4. `llm = ChatOpenAI(model=\"gpt-4o\", temperature=0)`.<br>5. `agent = create_react_agent(llm, tools)`.<br>6. `agent.invoke({\"input\": \"What is 15 plus 27?\"})`. |
| Test | Output includes 42 and final answer line. |
| Stretch | Swap in Ollama (`ChatOllama`). |

---

## 22  Multi-Server, Multi-Step Research Agent (LangChain only)
Similar to #6 but without LangGraph.

| Section | Details |
| Libs | same as #15 |
| Layout |```\nmulti_server_agent/\n└─ agent.py```|
| Plan | 1. Re-use search & notion servers or create lightweight stubs.<br>2. Load both via `MultiServerMCPClient`.<br>3. Build a **ReAct** agent with both tools.<br>4. Prompt engineering: include tool schema in system message.<br>5. Loop until LLM stops emitting tool calls. |
| Test | “Find latest AI article and save TL;DR to Notion” (dummy). |
| Stretch | Add retry logic on 429s. |

---

## 23  HTTP-SSE Deployment & Streaming Resource
Switches transport and streams large files.

| Section | Details |
| Libs | `pip install mcp fastapi sse-starlette` |
| Layout |```\nstreaming_server/\n├─ server.py\n└─ reports/annual_report.pdf```|
| Plan | **Server**<br>1. `mcp.run(transport=\"sse\", host=\"0.0.0.0\", port=5000)`.<br>2. `@mcp.resource(\"annual_report\")` → `open(\"reports/annual_report.pdf\", \"rb\")`.<br><br>**Client**<br>3. `session.connect(HttpServerParameters(\"https://127.0.0.1:5000\"))`.<br>4. `for chunk in session.read_resource(\"annual_report\", chunk_size=2048): ...`. |
| Test | Download file; diff with original. |
| Stretch | Add TLS via `uvicorn --ssl-keyfile …`. |

---

## 8 (b)  Chatbot Context Server & Client  *(duplicate rank in list)*
Provides preferences to an LLM.

| Section | Details |
| Libs | `pip install mcp openai` |
| Layout |```\nchat_context/\n├─ server.py\n└─ client.py```|
| Plan | **Server**:<br>1. Resource `get_prefs` returns dict with topic & tone.<br><br>**Client**:<br>2. Read prefs with `read_resource`.<br>3. Build prompt: f\"User prefers {tone} on {topic}. {question}\".<br>4. Call `ChatCompletion`. |
| Test | Ask “Explain GPUs” and observe formal tech answer. |
| Stretch | Store prefs in SQLite and update via another tool. |

---

## 9  Ollama Client with MCP Context
Local model instead of OpenAI.

| Section | Details |
| Libs | `pip install requests mcp` |
| Plan | 1. Re-use server #2 or #8b.<br>2. Fetch context via MCP.<br>3. `requests.post(\"http://localhost:11434/api/chat\", json={...})` with context in messages.<br>4. Stream response. |
| Stretch | Script CLI `--model llama3.2 --user alice`. |

---

## 10  Stock-Price + Sentiment Bundler
Two APIs + simple polarity score.

| Section | Details |
| Libs | `pip install mcp yfinance newsapi-python textblob` |
| Plan | 1. Tool `get_stock_insights(ticker)`:<br>   * `price = yfinance.Ticker(ticker).info[\"currentPrice\"]`.<br>   * Use NewsAPI to pull latest 5 headlines.<br>   * `sentiment = TextBlob(...).sentiment.polarity` average.<br>   * Return dict. |
| Stretch | Cache to avoid rate limits. |

---

## 11  PostgreSQL Query Server
Read-only DB access.

| Section | Details |
| Libs | `pip install mcp asyncpg` |
| Plan | 1. `pool = await asyncpg.create_pool(...)` in startup.<br>2. `@mcp.tool() async def workspace_rows(query:str)->list[dict]`.<br>3. Guard: `if not re.match(r\"^\\s*SELECT\", query, re.I): raise ValueError`. |
| Stretch | Add limit-100 auto-wrap. |

---

## 12  Notion Page Creator
Write-back workflow.

| Section | Details |
| Libs | `pip install mcp notion-client python-dotenv` |
| Plan | 1. Read `NOTION_API_KEY` from `.env`.<br>2. Tool `create_notion_page(title, content)`.<br>3. Return page URL on success. |
| Stretch | Support markdown → Notion blocks conversion. |

---

## 13  Secure MCP Server (API Key Auth)
Adds simple auth field.

| Section | Details |
| Plan | 1. Extend calculator (#1).<br>2. Tool signature `add(a:int, b:int, api_key:str)`.<br>3. `if api_key != os.getenv(\"SECRET_KEY\"): raise PermissionError`.<br>4. Client must pass key. |
| Stretch | Move check into decorator for DRY. |

---

## 14  Production-Ready MCP Server (Lifespan)
Shows startup/shutdown hooks.

| Section | Details |
| Libs | `pip install mcp` |
| Plan | 1. `async def startup(): acquire pool`.<br>2. `async def shutdown(): pool.close()`.<br>3. Use `async with lifespan(startup, shutdown): mcp.run(...)`. |
| Stretch | Emit Prometheus metrics. |

---

## 16  RAG Enhancement with MCP Real-time Data
Vector store + live quote.

| Section | Details |
| Libs | `pip install langchain langchain-openai faiss-cpu mcp` |
| Plan | 1. Server exposes `/stocks/{ticker}` resource (random price or real API).<br>2. In RAG pipeline hook, check query for REGEX ticker; fetch via MCP.<br>3. Append fetched price to doc context before LLM call. |
| Stretch | Use actual AlphaVantage prices. |

---

## 17  Simple LangGraph Workflow Calling an MCP Tool
First LangGraph.

| Section | Details |
| Libs | `pip install langgraph langchain langchain-mcp-adapters` |
| Plan | 1. Load `add` tool via adapter.<br>2. `WorkflowState` with `a,b,sum`.<br>3. Node `compute_sum` → call `add` tool, update state.<br>4. Compile; invoke with `a=7,b=5`. |
| Stretch | Parallel branch that logs before END. |

---

## 19  Local Llama 2 Agent (Ollama + LangChain)
Offline LLM.

| Section | Details |
| Libs | `pip install langchain langchain-ollama langchain-mcp-adapters` |
| Plan | 1. Start Ollama `ollama run llama2` in background.<br>2. Same steps as #15 but `llm = ChatOllama(model=\"llama2\")`. |
| Stretch | Quantised 7B vs 70B benchmark. |

---

## 20  PR Assistant End-to-End (Client + Server)
Full CLI loop.

| Section | Details |
| Libs | `pip install mcp langchain-openai anthropic` |
| Plan | 1. Re-use GitHub PR Fetcher server (#18 from original list) or stub.<br>2. Client spawns server subprocess.<br>3. Build tool schema JSON; send to Anthropic chat.<br>4. Regex for `{\"tool\": …}` → `session.run_tool()`.<br>5. Loop until model stops emitting tool calls. |
| Stretch | Use parallel Notion log via #12. |

---

## 21  LangGraph with External MCP State Management
Two-way state get/set.

| Section | Details |
| Libs | `pip install langgraph langchain-mcp-adapters` |
| Plan | **Server**: `get_state`, `set_state` tools + in-mem dict.<br>**Graph**:<br>1. Node `load_state` → call `get_state`.<br>2. Node `process` – mutate.<br>3. Node `save_state` → call `set_state`.<br>4. END. |
| Stretch | Replace dict with Redis. |

---

### Final tips

* **Create a mono-repo** with one folder per rank so you can `git tag` each milestone and link straight from video descriptions.  
* **Start every project with a 30-second smoke test** (the “Tests” section) so viewers see success fast.  
* Whenever a project *re-uses* code from an earlier rank, import it as a library (`from basic_calculator_server import add`) to avoid duplication on screen.