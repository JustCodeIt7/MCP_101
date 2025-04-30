from mcp.server.fastmcp import FastMCP

# Instantiate FastMCP
mcp = FastMCP("user_profile_server")

# Mock user data
USERS = {
    "alice": {
        "name": "Alice Smith",
        "email": "alice@example.com",
        "age": 28,
        "interests": ["AI", "Machine Learning", "Python"]
    },
    "bob": {
        "name": "Bob Johnson",
        "email": "bob@example.com",
        "age": 34,
        "interests": ["Data Science", "Cloud Computing", "JavaScript"]
    },
    "charlie": {
        "name": "Charlie Brown",
        "email": "charlie@example.com",
        "age": 22,
        "interests": ["Web Development", "Mobile Apps", "Gaming"]
    }
}

@mcp.resource("users://{user_id}/profile")
def get_user_profile(user_id: str):
    """Get a user's profile by their ID."""
    if user_id in USERS:
        return USERS[user_id]
    else:
        return {"error": f"User {user_id} not found"}

@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b

if __name__ == "__main__":
    mcp.run(transport="stdio")