from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Optional, List
import uvicorn
from core.chat.enhanced_chat import EnhancedChat
from core.memory.memory_store import MemoryStore
from core.prompt.prompt_manager import PromptManager

app = FastAPI(title="MiaOllama API")

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化核心组件
memory_store = MemoryStore()
prompt_manager = PromptManager()
chat_manager = EnhancedChat(
    memory_store=memory_store,
    prompt_manager=prompt_manager
)

class ChatRequest(BaseModel):
    conversation_id: str
    message: str
    user_preferences: Optional[Dict] = None
    stream: bool = False

class PromptTemplate(BaseModel):
    name: str
    content: Dict

@app.post("/api/chat")
async def chat(request: ChatRequest):
    """聊天接口"""
    try:
        response = chat_manager.chat(
            conversation_id=request.conversation_id,
            message=request.message,
            user_preferences=request.user_preferences,
            stream=request.stream
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/history/{conversation_id}")
async def get_history(conversation_id: str, limit: int = 10):
    """获取对话历史"""
    try:
        history = chat_manager.get_conversation_history(
            conversation_id=conversation_id,
            limit=limit
        )
        return {"history": history}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/prompts")
async def list_prompts():
    """列出所有提示词模板"""
    return {"prompts": prompt_manager.list_prompts()}

@app.post("/api/prompts")
async def add_prompt(prompt: PromptTemplate):
    """添加新的提示词模板"""
    try:
        prompt_manager.add_prompt(prompt.name, prompt.content)
        return {"message": "提示词模板添加成功"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/prompts/{name}")
async def delete_prompt(name: str):
    """删除提示词模板"""
    if prompt_manager.delete_prompt(name):
        return {"message": "提示词模板删除成功"}
    raise HTTPException(status_code=404, detail="提示词模板不存在")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
