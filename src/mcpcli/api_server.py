import json
from typing import List, Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

from mcpcli.chat_api import initialize_servers, chat_completion

# Global variable to store server streams
server_streams = []
context_managers = []

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    global server_streams, context_managers
    
    # Startup
    config_file = r"G:\Projects\mcp-via-litellm\server_config.json"
    try:
        with open(config_file, 'r') as f:
            servers = list(json.load(f)['mcpServers'].keys())
        
        # Initialize server connections
        streams, managers = await initialize_servers(config_file, servers)
        if not streams:
            raise RuntimeError("Failed to initialize servers")
        
        server_streams = streams
        context_managers = managers
        
        yield
    except Exception as e:
        raise RuntimeError(f"Startup failed: {str(e)}")
    finally:
        # Shutdown
        for cm in context_managers:
            try:
                await cm.__aexit__(None, None, None)
            except:
                pass

# Initialize FastAPI with lifespan
app = FastAPI(title="MCP Chat API", lifespan=lifespan)

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    add_system_prompt: Optional[bool] = False
    provider: Optional[str] = None
    model: Optional[str] = None

@app.post("/chat")
async def chat(request: ChatRequest):
    """Handle chat completion requests."""
    if not server_streams:
        raise HTTPException(status_code=503, detail="Server connections not initialized")

    try:
        # Convert Pydantic models to dictionaries
        messages = [msg.model_dump() for msg in request.messages]
        
        # Process chat completion
        chat_history = await chat_completion(
            server_streams=server_streams,
            messages=messages,
            add_system_prompt=request.add_system_prompt,
            provider=request.provider,
            model=request.model
        )
        
        if not chat_history:
            raise HTTPException(status_code=500, detail="Chat completion failed")
        
        return {"messages": chat_history}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def start_server(host: str = "0.0.0.0", port: int = 8132):
    """Start the FastAPI server."""
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    start_server()
