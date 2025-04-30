from mcp.server.fastmcp import FastMCP

# Instantiate FastMCP
mcp = FastMCP("chat_context_server")

# User preferences database (in-memory for simplicity)
USER_PREFS = {
    "default": {
        "topic": "technology",
        "tone": "formal",
        "detail_level": "medium",
        "examples": True
    },
    "alice": {
        "topic": "artificial intelligence",
        "tone": "technical",
        "detail_level": "high",
        "examples": True
    },
    "bob": {
        "topic": "data science",
        "tone": "casual",
        "detail_level": "medium",
        "examples": True
    },
    "charlie": {
        "topic": "web development",
        "tone": "enthusiastic",
        "detail_level": "low",
        "examples": False
    }
}

@mcp.resource("prefs://{user_id}")
def get_prefs(user_id: str):
    """
    Get the preferences for a specific user.
    
    Args:
        user_id: The ID of the user
        
    Returns:
        A dictionary containing the user's preferences
    """
    # Return the user's preferences if they exist, otherwise return default preferences
    return USER_PREFS.get(user_id, USER_PREFS["default"])

@mcp.tool()
def list_users() -> list:
    """
    List all available users in the system.
    
    Returns:
        A list of user IDs
    """
    return list(USER_PREFS.keys())

@mcp.tool()
def update_preference(user_id: str, preference: str, value: str) -> str:
    """
    Update a specific preference for a user.
    
    Args:
        user_id: The ID of the user
        preference: The preference to update (topic, tone, detail_level, examples)
        value: The new value for the preference
        
    Returns:
        A confirmation message
    """
    # Check if the user exists
    if user_id not in USER_PREFS:
        # Create a new user with default preferences
        USER_PREFS[user_id] = USER_PREFS["default"].copy()
    
    # Check if the preference is valid
    if preference not in ["topic", "tone", "detail_level", "examples"]:
        return f"Error: Invalid preference '{preference}'. Valid preferences are: topic, tone, detail_level, examples"
    
    # Handle boolean conversion for 'examples'
    if preference == "examples":
        if value.lower() in ["true", "yes", "1"]:
            USER_PREFS[user_id][preference] = True
        elif value.lower() in ["false", "no", "0"]:
            USER_PREFS[user_id][preference] = False
        else:
            return f"Error: Invalid value for 'examples'. Use 'true' or 'false'"
    else:
        # Update the preference
        USER_PREFS[user_id][preference] = value
    
    return f"Successfully updated {preference} to '{value}' for user {user_id}"

if __name__ == "__main__":
    mcp.run(transport="stdio")