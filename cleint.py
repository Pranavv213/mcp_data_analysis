import asyncio
from contextlib import asynccontextmanager
import os
import re
from typing import Dict, List, Optional
from pydantic import BaseModel

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

# LangChain / LangGraph imports
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import SystemMessage

# Prebuilt agent and structural state class tracking
from langgraph.prebuilt import create_react_agent
from langgraph.prebuilt.chat_agent_executor import AgentState as LangGraphAgentState


# 1. Inherit from LangGraph's Base AgentState to automatically include 'remaining_steps'
class AgentState(LangGraphAgentState):
    data_loaded: bool
    columns: list[str]
    dataset_path: Optional[str]


class ApplicationContext:
    def __init__(self):
        self.mcp_client: Optional[MultiServerMCPClient] = None
        self.tools: List = []
        self.agent_executor = None
        self.memory = MemorySaver()

    def dynamic_prompt_builder(self, state: AgentState) -> list:
        """
        Runs dynamically on every turn, injecting persistent state parameters 
        from the checkpointer thread directly into the LLM system prompt context.
        """
        data_loaded = state.get("data_loaded", False)
        columns = state.get("columns", [])
        dataset_path = state.get("dataset_path", "None")
        
        system_prompt = f"""You are a data analysis assistant with comprehensive visualization capabilities.

**CURRENT SESSION TRUTHS (MANAGED BY YOUR MEMORY CHECKPOINTER):**
- Data Loaded: {data_loaded}
- Target Dataset Path: {dataset_path}
- Columns: {', '.join(columns) if columns else 'Not loaded yet'}

**CRITICAL MEMORY DIRECTION:**
You have a persistent thread memory. If "Data Loaded" above is True, the file is already active in memory. DO NOT execute load_csv() or ask the user to load it again unless they explicitly request a completely new file.

**IMPORTANT WORKFLOW:**
1. When a user asks about data analysis, FIRST check if data is loaded.
2. If data is NOT loaded, suggest loading a CSV file using load_csv().
3. If data IS loaded, analyze the column names first using get_columns().

**AVAILABLE TOOLS:**
1. load_csv(path) - Load a CSV file from the filesystem.
2. get_columns() - Get all column names in the dataset
3. summary() - Get comprehensive dataset summary
4. average(column) - Calculate average of a numeric column
5. maximum(column) - Find maximum value in a column
6. plot_bar(x_column, y_column, title) - Create bar chart
7. plot_histogram(column, bins) - Create histogram
8. plot_scatter(x_column, y_column) - Create scatter plot
9. correlation_matrix() - Generate correlation heatmap
10. generate_full_visualization() - Generate ALL visualizations at once
11. create_dashboard() - Create comprehensive dashboard

**CURRENT STATUS:** {'✅ Data is ready for analysis and visualization!' if data_loaded else '⏳ No data loaded yet. Please load a CSV file first.'}"""

        return [SystemMessage(content=system_prompt)] + state["messages"]


ctx = ApplicationContext()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles startup MCP connection and agent assembly."""
    try:
        print("⏳ Connecting to MCP server...")
        ctx.mcp_client = MultiServerMCPClient(
            {
                "data-analysis": {
                    "command": "python",
                    "args": ["server.py"],
                    "transport": "stdio",
                    "env": {"PYTHONUNBUFFERED": "1"}
                }
            }
        )
        
        ctx.tools = await asyncio.wait_for(ctx.mcp_client.get_tools(), timeout=30.0)
        print("\n✅ Loaded Tools:")
        for tool in ctx.tools:
            print(f"  • {tool.name}")

        llm = ChatOllama(model="llama3.2:3b", temperature=0, timeout=60)
        
        # Uses prompt instead of state_modifier, passing our subclassed AgentState schema
        ctx.agent_executor = create_react_agent(
            model=llm,
            tools=ctx.tools,
            checkpointer=ctx.memory,
            prompt=ctx.dynamic_prompt_builder,  
            state_schema=AgentState
        )
        print("\n✅ LangGraph Thread-Controlled Agent Initialized Successfully!")
        yield
    except Exception as e:
        print(f"\n❌ Lifecycle Initialization Failed: {e}")
        raise


app = FastAPI(title="Data Analysis Agent API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str
    thread_id: str
    # NO path field here - it will be extracted from query

class QueryResponse(BaseModel):
    response: str
    thread_id: str
    data_loaded: bool
    columns: List[str]
    dataset_path: Optional[str] = None  # This will be "house.csv"


@app.post("/chat", response_model=QueryResponse)
async def process_agent_query(payload: QueryRequest):
    if not ctx.agent_executor:
        raise HTTPException(status_code=503, detail="Agent pipeline offline.")

    thread_id = payload.thread_id
    query = payload.query
    

    try:
        config = {"configurable": {"thread_id": thread_id}}
        
        # Pull state from checkpointer
        current_state = await ctx.agent_executor.aget_state(config)
        
        if not current_state.values:
            init_values = {
                "messages": [("user", query)],
                "data_loaded": False,
                "columns": [],
                "dataset_path": None
            }
        else:
            init_values = {"messages": [("user", query)]}

        # Invoke the workflow graph
        result = await ctx.agent_executor.ainvoke(init_values, config=config)
        assistant_message = result["messages"][-1].content
        
        # Extract filename from query - this will get "house.csv"
        filename = None
        
        # Look for filename with .csv extension
        csv_match = re.search(r'([a-zA-Z0-9_\-\.]+\.csv)', query, re.IGNORECASE)
        if csv_match:
            filename = csv_match.group(1)  # This will be "house.csv"
        else:
            # If no .csv found, look for filename after 'load'
            load_match = re.search(r'load(?:_csv)?\s+([a-zA-Z0-9_\-\.]+)', query, re.IGNORECASE)
            if load_match:
                potential_file = load_match.group(1)
                if potential_file.endswith('.csv'):
                    filename = potential_file
                else:
                    filename = f"{potential_file}.csv"
        
        # Update thread parameters in the checkpointer if tool outputs indicate file loaded successfully
        if "csv loaded successfully" in assistant_message.lower() or "successfully loaded" in assistant_message.lower():
            columns = []
            columns_match = re.search(r'Columns: (.*?)(?:\n|$)', assistant_message, re.IGNORECASE)
            if columns_match:
                columns = [c.strip() for c in columns_match.group(1).split(',') if c.strip()]
            
            # Set the path to the extracted filename (will be "house.csv")
            inferred_path = filename if filename else "unknown.csv"
            
            await ctx.agent_executor.aupdate_state(
                config,
                {
                    "data_loaded": True,
                    "columns": columns,
                    "dataset_path": inferred_path  # This will be "house.csv"
                }
            )

        # Retrieve the updated state values directly from the thread memory
        updated_state = await ctx.agent_executor.aget_state(config)
        
        return QueryResponse(
            response=assistant_message,
            thread_id=thread_id,
            data_loaded=updated_state.values.get("data_loaded", False),
            columns=updated_state.values.get("columns", []),
            dataset_path=updated_state.values.get("dataset_path", None)  # Returns "house.csv"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("cleint:app", host="0.0.0.0", port=8001, reload=False)
