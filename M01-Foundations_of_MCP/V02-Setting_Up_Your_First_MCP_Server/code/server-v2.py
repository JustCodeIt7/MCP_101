from mcp.server.fastmcp import FastMCP
import os
import smtplib
from dotenv import load_dotenv

load_dotenv()
# Create an MCP server
mcp = FastMCP("Email Tool", host="0.0.0.0", port=8000)

#MCP tool to send a mail
@mcp.tool()
def send_email(message: str, subject: str, reciever_email: str) -> int:
    """send an email to my email address"""
    sender_email = os.getenv("SENDER_EMAIL")
    text = f"Subject: {subject}\n\n{message}"
    server = smtplib.SMTP("smtp.gmail.com", 587)
    server.starttls()
    server.login(sender_email, os.getenv("GMAIL_APP_PASSOWORD"))
    server.sendmail(sender_email, reciever_email, text)
    return f"Email has been sent succesfully to {reciever_email}"



if __name__ == "__main__":
    mcp.run(transport="sse")
