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

@mcp.resource("users://all")
def get_all_users():
    """Get all users' profiles."""
    return USERS

@mcp.tool()
def add_user(user_id: str, name: str, email: str, age: int, interests: list) -> dict:
    """
    Add a new user to the database.
    
    Args:
        user_id: A unique identifier for the user
        name: The user's full name
        email: The user's email address
        age: The user's age
        interests: A list of the user's interests
        
    Returns:
        The newly created user profile or an error message
    """
    if user_id in USERS:
        return {"error": f"User {user_id} already exists"}
    
    USERS[user_id] = {
        "name": name,
        "email": email,
        "age": age,
        "interests": interests
    }
    
    return USERS[user_id]

@mcp.tool()
def remove_user(user_id: str) -> dict:
    """
    Remove a user from the database.
    
    Args:
        user_id: The unique identifier of the user to remove
        
    Returns:
        A success or error message
    """
    if user_id in USERS:
        user = USERS.pop(user_id)
        return {"success": f"User {user_id} ({user['name']}) removed successfully"}
    else:
        return {"error": f"User {user_id} not found"}

@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b

if __name__ == "__main__":
    mcp.run(transport="stdio")