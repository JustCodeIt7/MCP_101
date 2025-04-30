from mcp.server.fastmcp import FastMCP
from pathlib import Path
import os

# Instantiate FastMCP
mcp = FastMCP("secure_file_reader")

# Define the allowed root directory
ALLOWED_ROOT = Path("docs").resolve()

@mcp.tool()
def read_file(path: str) -> str:
    """
    Securely read a file from the allowed directory.
    
    Args:
        path: Relative path to the file within the allowed directory
        
    Returns:
        The content of the file (truncated to 5000 bytes if larger)
        
    Raises:
        PermissionError: If the path tries to access files outside the allowed directory
        FileNotFoundError: If the file doesn't exist
    """
    try:
        # Resolve the absolute path and ensure it's within the allowed root
        abs_path = (ALLOWED_ROOT / path).resolve()
        
        # Security check: ensure the path is within the allowed root
        if not abs_path.is_relative_to(ALLOWED_ROOT):
            raise PermissionError(f"Access denied: Cannot access files outside the allowed directory")
        
        # Check if the file exists
        if not abs_path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        
        # Check if it's a regular file (not a directory or special file)
        if not abs_path.is_file():
            raise ValueError(f"Not a file: {path}")
        
        # Read the file content
        with open(abs_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # Truncate if too large
        if len(content) > 5000:
            content = content[:5000] + "\n\n... [Content truncated due to size limit] ..."
        
        return content
    
    except Exception as e:
        # Convert any other exceptions to a readable error message
        return f"Error reading file: {str(e)}"

@mcp.tool()
def find_files(keyword: str = "") -> str:
    """
    Find files in the allowed directory that contain the given keyword in their name.
    If no keyword is provided, list all files.
    
    Args:
        keyword: Optional keyword to filter files by name
        
    Returns:
        A list of files as a string
    """
    try:
        # Create the docs directory if it doesn't exist
        if not ALLOWED_ROOT.exists():
            ALLOWED_ROOT.mkdir(parents=True)
        
        # Walk the directory tree
        found_files = []
        for root, dirs, files in os.walk(ALLOWED_ROOT):
            root_path = Path(root)
            # Skip hidden directories
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            
            for file in files:
                # Skip hidden files
                if file.startswith('.'):
                    continue
                
                # Check if the keyword is in the filename
                if keyword.lower() in file.lower():
                    # Get the relative path from the allowed root
                    rel_path = root_path.relative_to(ALLOWED_ROOT) / file
                    found_files.append(str(rel_path))
        
        if found_files:
            return "Found files:\n" + "\n".join(found_files)
        else:
            if keyword:
                return f"No files found containing '{keyword}'"
            else:
                return "No files found in the allowed directory"
    
    except Exception as e:
        return f"Error finding files: {str(e)}"

# Create a sample file in the docs directory for testing
def create_sample_files():
    """Create some sample files in the docs directory for testing."""
    sample_dir = ALLOWED_ROOT / "samples"
    sample_dir.mkdir(exist_ok=True)
    
    # Create a README file
    with open(ALLOWED_ROOT / "README.txt", "w") as f:
        f.write("""
# Secure File Reader

This is a sample README file for testing the secure file reader.

## Features
- Path sanitization to prevent directory traversal attacks
- File size truncation to prevent memory issues
- File search functionality
        """)
    
    # Create a sample text file
    with open(sample_dir / "sample.txt", "w") as f:
        f.write("This is a sample text file for testing the secure file reader.")
    
    # Create a sample Python file
    with open(sample_dir / "example.py", "w") as f:
        f.write("""
def hello_world():
    \"\"\"Print a greeting message.\"\"\"
    print("Hello, world!")

if __name__ == "__main__":
    hello_world()
        """)

if __name__ == "__main__":
    # Create sample files for testing
    create_sample_files()
    
    # Run the server
    mcp.run(transport="stdio")