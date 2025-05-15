from mcp.server.fastmcp import FastMCP
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider


from pydantic_ai import Agent

server = FastMCP('PydanticAI Server')

# Ollama Agent
ollama_model = OpenAIModel(
    model_name='llama3.2', provider=OpenAIProvider(base_url='http://eos-parkmour.netbird.cloud:11434/v1')
)
server_agent = Agent(ollama_model)

@server.tool()
async def poet(theme: str) -> str:
    """Poem generator"""
    r = await server_agent.run(f'write a poem about {theme}')
    return r.output

# --- 1. Define a Resource that returns "Hello, world!" ---
# Resources expose data to LLMs like GET endpoints in a REST API, providing information without complex computations or side effects.
@server.resource("hello://world")
def hello_resource() -> str:
    """Return a simple greeting."""
    # e.g. call an external API, or do some complex math
    content = "Hello, world!"
    # return the result
    return content

# --- 2. Prompt the user to enter a name ---
# Prompts are reusable templates that help LLMs interact with your server effectively:
@server.prompt()
def hello_prompt(name: str) -> str:
    """Prompt to greet a user by name."""
    # Prompt template to greet the user
    # This is a simple example, but you can use more complex templates
    prompt_template = f"Hello, {name}!"
    return prompt_template

# --- 3. Define a Tool that also returns "Hello, world!" ---
# Tools enable LLMs to execute actions via your server, performing computations and generating side effects beyond passive resource retrieval.
@server.tool()
async def hello_tool(name:str) -> str:
    """Tool that returns the same greeting."""
    p = f"Hello, world! {name}"
    r = await server_agent.run(p)
    return r.output

if __name__ == '__main__':
    server.run()