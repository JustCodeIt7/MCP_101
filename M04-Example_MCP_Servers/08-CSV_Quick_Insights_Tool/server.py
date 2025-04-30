from mcp.server.fastmcp import FastMCP
from pathlib import Path
import pandas as pd
import os

# Instantiate FastMCP
mcp = FastMCP("csv_insights")

# Define the allowed root directory (current directory)
ALLOWED_ROOT = Path(".").resolve()

@mcp.tool()
def csv_summary(path: str) -> str:
    """
    Generate a summary of a CSV file.
    
    Args:
        path: Path to the CSV file (must be within the allowed directory)
        
    Returns:
        A string containing summary statistics and a preview of the CSV data
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
        
        # Check if it's a CSV file
        if not path.lower().endswith('.csv'):
            raise ValueError(f"Not a CSV file: {path}")
        
        # Read the CSV file
        df = pd.read_csv(abs_path)
        
        # Build the summary
        summary = []
        
        # Basic info
        summary.append(f"## CSV Summary for: {path}")
        summary.append(f"Rows: {df.shape[0]}")
        summary.append(f"Columns: {df.shape[1]}")
        summary.append("")
        
        # Column names and types
        summary.append("## Columns")
        dtypes = df.dtypes.to_dict()
        for col, dtype in dtypes.items():
            summary.append(f"- {col}: {dtype}")
        summary.append("")
        
        # Data preview
        summary.append("## Data Preview (First 5 rows)")
        summary.append(df.head(5).to_markdown())
        summary.append("")
        
        # Summary statistics
        summary.append("## Summary Statistics")
        # Only include numeric columns in describe
        numeric_df = df.select_dtypes(include=['number'])
        if not numeric_df.empty:
            summary.append(numeric_df.describe().to_markdown())
        else:
            summary.append("No numeric columns found for statistics.")
        
        # Missing values
        summary.append("")
        summary.append("## Missing Values")
        missing = df.isnull().sum()
        if missing.sum() > 0:
            missing_data = [(col, count) for col, count in missing.items() if count > 0]
            if missing_data:
                for col, count in missing_data:
                    summary.append(f"- {col}: {count} missing values")
            else:
                summary.append("No missing values found.")
        else:
            summary.append("No missing values found.")
        
        return "\n".join(summary)
    
    except Exception as e:
        # Convert any exceptions to a readable error message
        return f"Error analyzing CSV file: {str(e)}"

# Create a sample CSV file for testing
def create_sample_csv():
    """Create a sample CSV file for testing."""
    # Create a sample directory if it doesn't exist
    sample_dir = Path("samples")
    sample_dir.mkdir(exist_ok=True)
    
    # Create a sample CSV file
    sample_path = sample_dir / "sales.csv"
    
    # Sample data
    data = """date,product,category,price,quantity,revenue
2023-01-01,Laptop,Electronics,1200.00,5,6000.00
2023-01-02,Smartphone,Electronics,800.00,10,8000.00
2023-01-03,Headphones,Electronics,150.00,20,3000.00
2023-01-04,Desk Chair,Furniture,250.00,8,2000.00
2023-01-05,Coffee Table,Furniture,350.00,5,1750.00
2023-01-06,Desk Lamp,Furniture,45.00,15,675.00
2023-01-07,Tablet,Electronics,500.00,12,6000.00
2023-01-08,Monitor,Electronics,300.00,7,2100.00
2023-01-09,Keyboard,Electronics,80.00,25,2000.00
2023-01-10,Mouse,Electronics,30.00,30,900.00
"""
    
    with open(sample_path, "w") as f:
        f.write(data)
    
    print(f"Created sample CSV file at: {sample_path}")
    return str(sample_path)

if __name__ == "__main__":
    # Create a sample CSV file for testing
    sample_csv = create_sample_csv()
    
    print(f"To test this server, call the csv_summary tool with path: {sample_csv}")
    
    # Run the server
    mcp.run(transport="stdio")