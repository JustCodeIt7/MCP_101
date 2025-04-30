from mcp.server.fastmcp import FastMCP

# Instantiate FastMCP
mcp = FastMCP("code_review")

@mcp.tool()
def analyze_complexity(code: str) -> str:
    """
    Analyze the complexity of the provided code.
    This is a dummy implementation that returns a simple analysis.
    """
    # In a real implementation, you might use tools like 'radon' to calculate
    # cyclomatic complexity, but for now we'll return a dummy response
    lines = code.strip().split("\n")
    line_count = len(lines)
    
    if line_count < 10:
        return "Low complexity: The code is short and likely simple."
    elif line_count < 50:
        return "Medium complexity: The code has a moderate number of lines."
    else:
        return "High complexity: The code is lengthy and might be difficult to maintain."

@mcp.prompt()
def review_code(code: str) -> str:
    """
    Generate a prompt template for code review.
    This helps guide an LLM to provide a structured code review.
    """
    return f"""
Please review the following code and provide feedback:

```python
{code}
```

Consider the following aspects in your review:
1. Code style and adherence to PEP 8
2. Potential bugs or edge cases
3. Performance considerations
4. Readability and maintainability
5. Suggestions for improvement

Your review:
"""

@mcp.resource("config")
def get_config():
    """
    Return a read-only configuration for the code review server.
    """
    return {
        "version": "1.0.0",
        "max_code_length": 5000,
        "supported_languages": ["python", "javascript", "java"],
        "review_guidelines": {
            "style": "Follow language-specific style guides",
            "performance": "Identify O(n²) or worse algorithms",
            "security": "Flag potential injection vulnerabilities"
        }
    }

if __name__ == "__main__":
    mcp.run(transport="stdio")