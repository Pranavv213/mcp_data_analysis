import asyncio

from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_ollama import ChatOllama


class DataAnalysisAgent:
    def __init__(self):
        self.data_loaded = False
        self.columns = []
        self.messages = []
        self.agent = None
        
    async def initialize(self):
        client = MultiServerMCPClient(
            {
                "data-analysis": {
                    "command": "python",
                    "args": ["server.py"],
                    "transport": "stdio",
                }
            }
        )
        
        tools = await client.get_tools()
        
        print("\nLoaded Tools:")
        for tool in tools:
            print(f"• {tool.name}")
        
        llm = ChatOllama(
            model="llama3.2:3b",
            temperature=0,
        )
        
        self.agent = create_agent(
            model=llm,
            tools=tools,
            system_prompt=f"""
You are a data analysis assistant with SESSION MEMORY.

**CURRENT SESSION STATE:**
- Data Loaded: {self.data_loaded}
- Columns: {self.columns if self.columns else 'Not loaded yet'}

**IMPORTANT CONTEXT:**
The data state is being tracked externally. Here's what you know:
{'✅ A CSV is already loaded with columns: ' + ', '.join(self.columns) if self.data_loaded else '❌ No CSV is loaded yet'}

**YOUR RULES:**
1. If data is loaded (as shown above), DO NOT ask the user to load it again
2. Use the tools to get actual data, but remember the context
3. For column queries, still call get_columns() to get the latest list
4. If data is already loaded, proceed with analysis without asking to load

**RESPONSE GUIDELINES:**
- If asked about columns: Call get_columns() and list them
- If asked for statistics: Verify column exists with get_columns(), then use the appropriate tool
- If data IS loaded: No need to suggest loading again
- If data is NOT loaded: Ask to load with load_csv()

Remember: {'Data is ready for analysis!' if self.data_loaded else 'Waiting for data to be loaded.'}
"""
        )
    
    async def process_query(self, query: str):
        # Add user message to history
        self.messages.append({"role": "user", "content": query})
        
        try:
            response = await self.agent.ainvoke({"messages": self.messages})
            assistant_message = response["messages"][-1]
            self.messages.append(assistant_message)
            
            # Update state based on response (check if load_csv was called)
            response_text = assistant_message.content.lower()
            if "csv loaded successfully" in response_text.lower():
                self.data_loaded = True
                # Try to extract columns from response
                try:
                    # Call get_columns to update columns
                    get_cols_tool = next((t for t in self.agent.tools if t.name == "get_columns"), None)
                    if get_cols_tool:
                        # Note: This might need adjustment based on your LangChain version
                        pass
                except:
                    pass
            
            return assistant_message.content
            
        except Exception as e:
            print(f"Error: {e}")
            self.messages.pop()
            return f"Error: {e}"


async def main():
    agent = DataAnalysisAgent()
    await agent.initialize()
    
    print("\n" + "="*50)
    print("🤖 DATA ANALYSIS AGENT READY")
    print("="*50)
    print("✓ Persistent session memory")
    print("✓ Remembers loaded data between questions")
    print("="*50)
    print("\nType 'exit' to quit.\n")
    
    while True:
        query = input("You: ")
        
        if query.lower() == "exit":
            break
        
        response = await agent.process_query(query)
        
        print("\nAssistant:")
        print(response)
        print()


if __name__ == "__main__":
    asyncio.run(main())