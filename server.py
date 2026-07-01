from fastmcp import FastMCP
import pandas as pd
import matplotlib.pyplot as plt
import os

mcp = FastMCP(
    name="Data Analysis MCP",
    instructions="""
    This server provides basic data analysis tools.
    First call load_csv() before using any other tool.

    
    """
)

# Global dataframe
df = None


@mcp.tool
def load_csv(path: str) -> str:
    """
    Load a CSV file.
    """

    global df

    try:
        df = pd.read_csv(path)

        return (
            f"CSV loaded successfully!\n\n"
            f"Rows: {len(df)}\n"
            f"Columns: {', '.join(df.columns)}"
        )

    except Exception as e:
        return f"Error: {e}"


@mcp.tool
def get_columns() -> list[str]:
    """
    Return all column names.
    """

    global df

    if df is None:
        return ["Please load a CSV first."]

    return df.columns.tolist()


@mcp.tool
def summary() -> dict:
    """
    Return dataset summary.
    """

    global df

    if df is None:
        return {"error": "Please load a CSV first."}

    return {
        "rows": len(df),
        "columns": len(df.columns),
        "column_names": df.columns.tolist(),
        "missing_values": df.isnull().sum().to_dict(),
        "numeric_columns": df.select_dtypes(include="number").columns.tolist(),
    }


@mcp.tool
def average(column: str):
    """
    Calculate average of a numeric column.
    """

    global df

    if df is None:
        return "Please load a CSV first."

    if column not in df.columns:
        return "Column not found."

    return float(df[column].mean())


@mcp.tool
def maximum(column: str):
    """
    Find maximum value in a column.
    """

    global df

    if df is None:
        return "Please load a CSV first."

    if column not in df.columns:
        return "Column not found."

    return df[column].max()


@mcp.tool
def plot_bar(x_column: str, y_column: str) -> str:
    """
    Generate a bar chart.
    """

    global df

    if df is None:
        return "Please load a CSV first."

    if x_column not in df.columns:
        return "X column not found."

    if y_column not in df.columns:
        return "Y column not found."

    plt.figure(figsize=(8,5))
    plt.bar(df[x_column], df[y_column])

    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.title(f"{y_column} vs {x_column}")

    output = "chart.png"

    plt.tight_layout()
    plt.savefig(output)
    plt.close()

    return os.path.abspath(output)


if __name__ == "__main__":
    mcp.run()