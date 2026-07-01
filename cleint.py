import asyncio
import os
import sys
from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_ollama import ChatOllama


class DataAnalysisAgent:
    def __init__(self):
        self.data_loaded = False
        self.columns = []
        self.dataset_path = None
        self.messages = []
        self.agent = None
        self.tools = []
        self.client = None
        
    async def initialize(self):
        """Initialize the agent with MCP server connection"""
        try:
            # Connect to MCP Server
            self.client = MultiServerMCPClient(
                {
                    "data-analysis": {
                        "command": "python",
                        "args": ["server.py"],
                        "transport": "stdio",
                        "env": {
                            "PYTHONUNBUFFERED": "1",
                        }
                    }
                }
            )
            
            print("⏳ Connecting to MCP server...")
            self.tools = await asyncio.wait_for(
                self.client.get_tools(),
                timeout=30.0
            )
            
            print("\n✅ Loaded Tools:")
            tool_names = []
            for tool in self.tools:
                print(f"  • {tool.name}")
                tool_names.append(tool.name)
            
            # Initialize LLM
            llm = ChatOllama(
                model="llama3.2:3b",
                temperature=0,
                timeout=60,
            )
            
            # Create agent
            self.agent = create_agent(
                model=llm,
                tools=self.tools,
                system_prompt=self._get_system_prompt()
            )
            
            print("\n✅ Agent initialized successfully!")
            
        except asyncio.TimeoutError:
            print("\n❌ Connection timeout: MCP server took too long to respond")
            raise
        except Exception as e:
            print(f"\n❌ Failed to initialize agent: {e}")
            raise
    
    def _get_system_prompt(self):
        """Generate dynamic system prompt based on current state"""
        return f"""
You are a data analysis assistant with comprehensive visualization capabilities.

**CURRENT SESSION STATE:**
- Data Loaded: {self.data_loaded}
- Columns: {', '.join(self.columns) if self.columns else 'Not loaded yet'}

**AVAILABLE TOOLS:**
1. load_csv() - Load a CSV file
2. get_columns() - Get all column names
3. summary() - Get dataset summary
4. average() - Calculate average of a column
5. maximum() - Find maximum value
6. plot_bar() - Create bar chart
7. plot_histogram() - Create histogram
8. plot_scatter() - Create scatter plot
9. correlation_matrix() - Generate correlation heatmap
10. generate_full_visualization() - Generate ALL visualizations at once
11. create_dashboard() - Create comprehensive dashboard

**VISUALIZATION COMMANDS:**
- "visualize all" or "generate full visualization" → Use generate_full_visualization()
- "show dashboard" → Use create_dashboard()  
- "plot x vs y" → Use plot_bar() or plot_scatter()
- "histogram of column" → Use plot_histogram()
- "correlation" → Use correlation_matrix()

**RESPONSE RULES:**
1. ALWAYS call get_columns() before suggesting columns
2. When user asks for visualization, suggest appropriate columns
3. For "visualize all", use generate_full_visualization()
4. Always show generated file paths

**REMEMBER:** 
{'Data is ready for analysis and visualization!' if self.data_loaded else 'Waiting for data to be loaded.'}
"""
    
    async def process_query(self, query: str):
        """Process user query with context awareness"""
        if not self.agent:
            return "Agent not initialized. Please restart."
        
        # Check for visualization commands
        lower_query = query.lower()
        if "visualize all" in lower_query or "generate full visualization" in lower_query or "show all charts" in lower_query:
            if not self.data_loaded:
                return "⚠️ No data loaded. Please load a CSV first using 'load data.csv'"
            # Force using the specific tool
            query = "Use generate_full_visualization() tool to create all visualizations"
        
        if "dashboard" in lower_query and ("show" in lower_query or "create" in lower_query or "generate" in lower_query):
            if not self.data_loaded:
                return "⚠️ No data loaded. Please load a CSV first using 'load data.csv'"
            query = "Use create_dashboard() tool to create a comprehensive dashboard"
        
        # Add user message to history
        self.messages.append({"role": "user", "content": query})
        
        try:
            # Process with current agent
            response = await self.agent.ainvoke({"messages": self.messages})
            assistant_message = response["messages"][-1]
            self.messages.append(assistant_message)
            
            # Parse response to update state
            response_text = assistant_message.content
            
            # Check if CSV was loaded successfully
            if "csv loaded successfully" in response_text.lower():
                self.data_loaded = True
                # Try to extract columns from response
                try:
                    import re
                    columns_match = re.search(r'Columns: (.*?)(?:\n|$)', response_text, re.IGNORECASE)
                    if columns_match:
                        cols_str = columns_match.group(1)
                        self.columns = [c.strip() for c in cols_str.split(',') if c.strip()]
                except:
                    pass
                
                # Recreate agent with updated state
                self.agent = create_agent(
                    model=self.agent.model,
                    tools=self.tools,
                    system_prompt=self._get_system_prompt()
                )
            
            return assistant_message.content
            
        except Exception as e:
            print(f"⚠️ Error processing query: {e}")
            self.messages.pop()
            return f"Error: {e}"


async def main():
    try:
        agent = DataAnalysisAgent()
        await agent.initialize()
        
        print("\n" + "="*70)
        print("🤖 DATA ANALYSIS AGENT WITH COMPREHENSIVE VISUALIZATION")
        print("="*70)
        print("\n📊 Commands:")
        print("  • 'load data.csv' - Load a CSV file")
        print("  • 'show columns' - List all columns")
        print("  • 'summary' - Get dataset summary")
        print("  • 'average age' - Calculate average")
        print("  • 'plot x vs y' - Create bar chart")
        print("  • 'histogram age' - Create histogram")
        print("  • 'correlation' - Show correlation matrix")
        print("  • 'visualize all' - Generate ALL visualizations!")
        print("  • 'show dashboard' - Create comprehensive dashboard")
        print("  • 'exit' - Quit the program")
        print("="*70)
        print("\n💡 Tip: Use 'visualize all' to generate comprehensive colorful visualizations!\n")
        
        while True:
            try:
                query = input("You: ")
                
                if query.lower() == "exit":
                    break
                
                if not query.strip():
                    continue
                
                response = await agent.process_query(query)
                
                print("\nAssistant:")
                print(response)
                print()
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n⚠️ Error: {e}")
                print("Please try again.\n")
                
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        print("\nTroubleshooting tips:")
        print("1. Make sure server.py is in the same directory")
        print("2. Check dependencies:")
        print("   pip install fastmcp pandas matplotlib seaborn langchain langchain-mcp-adapters langchain-ollama")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())