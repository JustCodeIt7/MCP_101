from mcp.server.fastmcp import FastMCP
import os
import smtplib
from dotenv import load_dotenv

load_dotenv()
# Create an MCP server
mcp = FastMCP("Email Tool", host="0.0.0.0", port=8000)

#MCP tool to send a mail
@mcp.tool()
def send_email(message: str, subject: str, reciever_email: str) -> str:
    """email my email address"""
    print(f'Sending email to {reciever_email}')
    sender_email = os.getenv("SENDER_EMAIL")
    text = f"Subject: {subject}\n\n{message}"
    server = smtplib.SMTP("smtp.gmail.com", 587)
    server.starttls()
    server.login(sender_email, os.getenv("GMAIL_APP_PASSOWORD"))
    server.sendmail(sender_email, reciever_email, text)
    return f"Email has been sent succesfully to {reciever_email}"

# --- 1. Define a Resource that returns "Hello, world!" ---
# Resources expose data to LLMs like GET endpoints in a REST API, providing information without complex computations or side effects.
@mcp.resource("hello://world")
def hello_resource() -> str:
    """Return a simple greeting."""
    print('Hello world resource called')
    return "Hello, world!"

# --- 2. Prompt the user to enter a name ---
# Prompts are reusable templates that help LLMs interact with your server effectively:
@mcp.prompt()
def hello_prompt(name: str) -> str:
    """Prompt to greet a user by name."""
    print('Hello world prompt called')
    prompt_template = f"Hello, {name}!"
    return prompt_template

# --- 3. Define a Tool that also returns "Hello, world!" ---
# Tools enable LLMs to execute actions via your server, performing computations and generating side effects beyond passive resource retrieval.
@mcp.tool()
def hello_tool(name:str) -> str:
    """tool that returns greeting with name"""
    print('Hello world tool called')
    return f"Hello, world! {name}"

if __name__ == "__main__":
    mcp.run(transport="sse")