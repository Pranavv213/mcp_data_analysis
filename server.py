from fastmcp import FastMCP
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import numpy as np
from datetime import datetime

# Use non-interactive backend
import matplotlib
matplotlib.use('Agg')

# Set style for beautiful charts
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

mcp = FastMCP(
    name="Data Analysis MCP",
    instructions="""
    This server provides comprehensive data analysis and visualization.
    First call load_csv() before using any other tool.
    """
)

# Global dataframe
df = None

@mcp.tool
def load_csv(path: str) -> str:
    """Load a CSV file."""
    global df
    try:
        if not os.path.exists(path):
            return f"❌ Error: File '{path}' not found."
        
        df = pd.read_csv(path)
        
        # Get data types info
        dtype_info = df.dtypes.astype(str).to_dict()
        
        return (
            f"✅ CSV loaded successfully!\n\n"
            f"📊 Rows: {len(df)}\n"
            f"📋 Columns: {', '.join(df.columns)}\n"
            f"📝 Data Types: {dtype_info}"
        )
    except Exception as e:
        return f"❌ Error loading CSV: {str(e)}"

@mcp.tool
def get_columns() -> list:
    """Return all column names."""
    global df
    if df is None:
        return ["Please load a CSV first."]
    return df.columns.tolist()

@mcp.tool
def summary() -> dict:
    """Return dataset summary."""
    global df
    if df is None:
        return {"error": "Please load a CSV first."}
    
    try:
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        categorical_cols = df.select_dtypes(include="object").columns.tolist()
        
        return {
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": df.columns.tolist(),
            "missing_values": df.isnull().sum().to_dict(),
            "numeric_columns": numeric_cols,
            "categorical_columns": categorical_cols,
            "data_types": df.dtypes.astype(str).to_dict(),
        }
    except Exception as e:
        return {"error": f"Error generating summary: {str(e)}"}

@mcp.tool
def average(column: str):
    """Calculate average of a numeric column."""
    global df
    if df is None:
        return "Please load a CSV first."
    if column not in df.columns:
        return f"Column '{column}' not found."
    try:
        if not pd.api.types.is_numeric_dtype(df[column]):
            return f"Column '{column}' is not numeric."
        return float(df[column].mean())
    except Exception as e:
        return f"Error calculating average: {str(e)}"

@mcp.tool
def maximum(column: str):
    """Find maximum value in a column."""
    global df
    if df is None:
        return "Please load a CSV first."
    if column not in df.columns:
        return f"Column '{column}' not found."
    try:
        if not pd.api.types.is_numeric_dtype(df[column]):
            return f"Column '{column}' is not numeric."
        return df[column].max()
    except Exception as e:
        return f"Error finding maximum: {str(e)}"

@mcp.tool
def plot_bar(x_column: str, y_column: str, title: str = None) -> str:
    """Generate a bar chart."""
    global df
    if df is None:
        return "Please load a CSV first."
    if x_column not in df.columns:
        return f"X column '{x_column}' not found."
    if y_column not in df.columns:
        return f"Y column '{y_column}' not found."
    
    try:
        plt.figure(figsize=(12, 7))
        
        # Create colorful bar chart
        colors = plt.cm.Set3(np.linspace(0, 1, min(len(df), 20)))
        
        if len(df) > 100:
            grouped = df.groupby(x_column)[y_column].mean().sort_values(ascending=False).head(20)
            bars = plt.bar(grouped.index.astype(str), grouped.values, color=colors, edgecolor='black', linewidth=1)
            plt.ylabel(f"Average {y_column}")
            title = title or f"Average {y_column} by {x_column} (Top 20)"
        else:
            bars = plt.bar(df[x_column].astype(str), df[y_column], color=colors, edgecolor='black', linewidth=1)
            plt.ylabel(y_column)
            title = title or f"{y_column} vs {x_column}"
        
        plt.xlabel(x_column)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=9)
        
        output = f"chart_{x_column}_{y_column}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(output, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return f"✅ Bar chart generated: {os.path.abspath(output)}"
    except Exception as e:
        return f"❌ Error generating chart: {str(e)}"

@mcp.tool
def plot_histogram(column: str, bins: int = 30) -> str:
    """Generate a histogram for a numeric column."""
    global df
    if df is None:
        return "Please load a CSV first."
    if column not in df.columns:
        return f"Column '{column}' not found."
    
    try:
        plt.figure(figsize=(12, 7))
        
        # Remove NaN values
        data = df[column].dropna()
        
        if len(data) == 0:
            return f"Column '{column}' has no valid data."
        
        # Create colorful histogram
        n, bins_edges, patches = plt.hist(data, bins=bins, edgecolor='black', linewidth=1.5, alpha=0.7)
        
        # Color each bar differently
        colors = plt.cm.viridis(np.linspace(0, 1, len(patches)))
        for patch, color in zip(patches, colors):
            patch.set_facecolor(color)
        
        plt.xlabel(column, fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title(f'Distribution of {column}', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        output = f"histogram_{column}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(output, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return f"✅ Histogram generated: {os.path.abspath(output)}"
    except Exception as e:
        return f"❌ Error generating histogram: {str(e)}"

@mcp.tool
def plot_scatter(x_column: str, y_column: str) -> str:
    """Generate a scatter plot."""
    global df
    if df is None:
        return "Please load a CSV first."
    if x_column not in df.columns:
        return f"X column '{x_column}' not found."
    if y_column not in df.columns:
        return f"Y column '{y_column}' not found."
    
    try:
        plt.figure(figsize=(12, 8))
        
        # Create colorful scatter plot
        scatter = plt.scatter(df[x_column], df[y_column], 
                            c=df[x_column], cmap='viridis', 
                            alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
        
        plt.colorbar(scatter, label=x_column)
        plt.xlabel(x_column, fontsize=12)
        plt.ylabel(y_column, fontsize=12)
        plt.title(f'Relationship: {y_column} vs {x_column}', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        output = f"scatter_{x_column}_{y_column}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(output, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return f"✅ Scatter plot generated: {os.path.abspath(output)}"
    except Exception as e:
        return f"❌ Error generating scatter plot: {str(e)}"

@mcp.tool
def generate_full_visualization() -> str:
    """Generate a comprehensive visualization showing all columns."""
    global df
    if df is None:
        return "Please load a CSV first."
    
    try:
        # Create a timestamp for unique filenames
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Determine column types
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        categorical_cols = df.select_dtypes(include="object").columns.tolist()
        
        # If no categorical columns, use first column as categorical
        if not categorical_cols and len(df.columns) > 0:
            categorical_cols = [df.columns[0]]
        
        generated_files = []
        
        # 1. Generate bar charts for categorical vs numeric
        if categorical_cols and numeric_cols:
            for cat_col in categorical_cols[:2]:  # Limit to first 2 categorical columns
                for num_col in numeric_cols[:2]:  # Limit to first 2 numeric columns
                    result = plot_bar(cat_col, num_col)  # REMOVED await
                    if "✅" in result:
                        generated_files.append(result.split(": ")[-1].strip())
        
        # 2. Generate histograms for numeric columns
        for num_col in numeric_cols:
            result = plot_histogram(num_col)  # REMOVED await
            if "✅" in result:
                generated_files.append(result.split(": ")[-1].strip())
        
        # 3. Generate scatter plots for numeric pairs
        if len(numeric_cols) >= 2:
            for i in range(min(2, len(numeric_cols)-1)):
                result = plot_scatter(numeric_cols[i], numeric_cols[i+1])  # REMOVED await
                if "✅" in result:
                    generated_files.append(result.split(": ")[-1].strip())
        
        # 4. Create a correlation heatmap
        if len(numeric_cols) >= 2:
            result = correlation_matrix()
            if "✅" in result:
                generated_files.append(result.split(": ")[-1].strip())
        
        # 5. Create a comprehensive dashboard
        dashboard_path = create_dashboard()
        generated_files.append(dashboard_path)
        
        return f"""
✅ Comprehensive visualization complete!

Generated {len(generated_files)} visualizations:

📊 Visualizations created:
{chr(10).join([f'  • {os.path.basename(f)}' for f in generated_files])}

📁 All files saved to: {os.path.dirname(generated_files[0]) if generated_files else os.getcwd()}

💡 To view the files:
   - Open them directly in your file explorer
   - The dashboard shows an overview of all visualizations
"""
    except Exception as e:
        return f"❌ Error generating full visualization: {str(e)}"

@mcp.tool
def correlation_matrix() -> str:
    """Generate a correlation matrix heatmap for numeric columns."""
    global df
    if df is None:
        return "Please load a CSV first."
    
    numeric_df = df.select_dtypes(include="number")
    if len(numeric_df.columns) < 2:
        return "Need at least 2 numeric columns for correlation."
    
    try:
        plt.figure(figsize=(12, 10))
        
        corr = numeric_df.corr()
        
        # Create a beautiful heatmap
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, cmap='coolwarm', 
                   center=0, fmt='.2f', square=True,
                   linewidths=0.5, cbar_kws={"shrink": 0.8})
        
        plt.title('Correlation Matrix', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        output = f"correlation_matrix_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(output, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return f"✅ Correlation matrix generated: {os.path.abspath(output)}"
    except Exception as e:
        return f"❌ Error generating correlation matrix: {str(e)}"

@mcp.tool
def create_dashboard() -> str:
    """Create a comprehensive dashboard with all visualizations."""
    global df
    if df is None:
        return "Please load a CSV first."
    
    try:
        # Create a figure with subplots
        fig = plt.figure(figsize=(20, 15))
        
        # Determine column types
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        categorical_cols = df.select_dtypes(include="object").columns.tolist()
        
        # If no categorical columns, use first column
        if not categorical_cols and len(df.columns) > 0:
            categorical_cols = [df.columns[0]]
        
        # Create subplot grid
        if len(numeric_cols) >= 2 and categorical_cols:
            # Subplot 1: Bar chart
            ax1 = plt.subplot(3, 3, 1)
            cat_col = categorical_cols[0]
            num_col = numeric_cols[0]
            if len(df) > 100:
                grouped = df.groupby(cat_col)[num_col].mean().sort_values(ascending=False).head(10)
                bars = ax1.bar(grouped.index.astype(str), grouped.values, 
                              color=plt.cm.Set3(np.linspace(0, 1, len(grouped))))
                ax1.set_ylabel(f"Average {num_col}")
                title = f"Top 10 {num_col} by {cat_col}"
            else:
                bars = ax1.bar(df[cat_col].astype(str), df[num_col],
                              color=plt.cm.Set3(np.linspace(0, 1, len(df))))
                ax1.set_ylabel(num_col)
                title = f"{num_col} vs {cat_col}"
            ax1.set_xlabel(cat_col)
            ax1.set_title(title, fontweight='bold')
            ax1.tick_params(axis='x', rotation=45)
            
            # Subplot 2: Second bar chart
            ax2 = plt.subplot(3, 3, 2)
            if len(categorical_cols) > 1 and len(numeric_cols) > 1:
                cat_col2 = categorical_cols[1]
                num_col2 = numeric_cols[1]
                if len(df) > 100:
                    grouped = df.groupby(cat_col2)[num_col2].mean().sort_values(ascending=False).head(10)
                    ax2.bar(grouped.index.astype(str), grouped.values, 
                           color=plt.cm.Paired(np.linspace(0, 1, len(grouped))))
                    ax2.set_ylabel(f"Average {num_col2}")
                    title = f"Top 10 {num_col2} by {cat_col2}"
                else:
                    ax2.bar(df[cat_col2].astype(str), df[num_col2],
                           color=plt.cm.Paired(np.linspace(0, 1, len(df))))
                    ax2.set_ylabel(num_col2)
                    title = f"{num_col2} vs {cat_col2}"
                ax2.set_xlabel(cat_col2)
                ax2.set_title(title, fontweight='bold')
                ax2.tick_params(axis='x', rotation=45)
        
        # Subplot 3: Histogram
        ax3 = plt.subplot(3, 3, 3)
        if numeric_cols:
            num_col = numeric_cols[0]
            data = df[num_col].dropna()
            n, bins, patches = ax3.hist(data, bins=30, edgecolor='black', linewidth=1, alpha=0.7)
            colors = plt.cm.viridis(np.linspace(0, 1, len(patches)))
            for patch, color in zip(patches, colors):
                patch.set_facecolor(color)
            ax3.set_xlabel(num_col)
            ax3.set_ylabel('Frequency')
            ax3.set_title(f'Distribution of {num_col}', fontweight='bold')
            ax3.grid(True, alpha=0.3)
        
        # Subplot 4: Second histogram
        ax4 = plt.subplot(3, 3, 4)
        if len(numeric_cols) > 1:
            num_col2 = numeric_cols[1]
            data = df[num_col2].dropna()
            n, bins, patches = ax4.hist(data, bins=30, edgecolor='black', linewidth=1, alpha=0.7)
            colors = plt.cm.plasma(np.linspace(0, 1, len(patches)))
            for patch, color in zip(patches, colors):
                patch.set_facecolor(color)
            ax4.set_xlabel(num_col2)
            ax4.set_ylabel('Frequency')
            ax4.set_title(f'Distribution of {num_col2}', fontweight='bold')
            ax4.grid(True, alpha=0.3)
        
        # Subplot 5: Scatter plot
        ax5 = plt.subplot(3, 3, 5)
        if len(numeric_cols) >= 2:
            scatter = ax5.scatter(df[numeric_cols[0]], df[numeric_cols[1]],
                                 c=df[numeric_cols[0]], cmap='viridis',
                                 alpha=0.6, s=30, edgecolors='black', linewidth=0.5)
            ax5.set_xlabel(numeric_cols[0])
            ax5.set_ylabel(numeric_cols[1])
            ax5.set_title(f'{numeric_cols[0]} vs {numeric_cols[1]}', fontweight='bold')
            ax5.grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=ax5, label=numeric_cols[0])
        
        # Subplot 6: Box plot
        ax6 = plt.subplot(3, 3, 6)
        if numeric_cols:
            data_to_plot = [df[col].dropna() for col in numeric_cols[:3]]
            bp = ax6.boxplot(data_to_plot, patch_artist=True)
            for patch, color in zip(bp['boxes'], plt.cm.Set3(np.linspace(0, 1, len(data_to_plot)))):
                patch.set_facecolor(color)
            ax6.set_xticklabels(numeric_cols[:3])
            ax6.set_title('Box Plots', fontweight='bold')
            ax6.grid(True, alpha=0.3)
        
        # Subplot 7: Correlation heatmap (small)
        ax7 = plt.subplot(3, 3, 7)
        if len(numeric_cols) >= 2:
            corr = df[numeric_cols].corr()
            im = ax7.imshow(corr, cmap='coolwarm', aspect='auto')
            ax7.set_xticks(range(len(corr.columns)))
            ax7.set_yticks(range(len(corr.columns)))
            ax7.set_xticklabels(corr.columns, rotation=45, ha='right')
            ax7.set_yticklabels(corr.columns)
            ax7.set_title('Correlation Matrix', fontweight='bold')
            plt.colorbar(im, ax=ax7)
        
        # Subplot 8: Pie chart (if categorical)
        ax8 = plt.subplot(3, 3, 8)
        if categorical_cols:
            cat_col = categorical_cols[0]
            value_counts = df[cat_col].value_counts().head(8)
            if len(value_counts) > 1:
                colors = plt.cm.Set3(np.linspace(0, 1, len(value_counts)))
                ax8.pie(value_counts.values, labels=value_counts.index, 
                       autopct='%1.1f%%', colors=colors, startangle=90)
                ax8.set_title(f'Distribution of {cat_col}', fontweight='bold')
        
        # Subplot 9: Line plot (if numeric)
        ax9 = plt.subplot(3, 3, 9)
        if len(numeric_cols) >= 2:
            # Sort by first numeric column for line plot
            sorted_df = df.sort_values(numeric_cols[0])
            x = sorted_df[numeric_cols[0]]
            y = sorted_df[numeric_cols[1]]
            ax9.plot(x, y, 'o-', color='blue', markersize=4, alpha=0.7)
            ax9.set_xlabel(numeric_cols[0])
            ax9.set_ylabel(numeric_cols[1])
            ax9.set_title(f'Trend: {numeric_cols[0]} vs {numeric_cols[1]}', fontweight='bold')
            ax9.grid(True, alpha=0.3)
        
        plt.suptitle(f'Data Analysis Dashboard - {len(df)} rows, {len(df.columns)} columns', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        output = f"dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(output, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return os.path.abspath(output)
    except Exception as e:
        return f"❌ Error creating dashboard: {str(e)}"

if __name__ == "__main__":
    try:
        mcp.run()
    except Exception as e:
        print(f"Server error: {e}", file=sys.stderr)
        sys.exit(1)
