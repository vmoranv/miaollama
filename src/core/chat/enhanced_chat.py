import requests
from typing import Dict, List, Optional
import json
from ..memory.memory_store import MemoryStore
from ..prompt.prompt_manager import PromptManager

class EnhancedChat:
    def __init__(
        self,
        ollama_base_url: str = "http://localhost:11434",
        model: str = "llama2",
        memory_store: Optional[MemoryStore] = None,
        prompt_manager: Optional[PromptManager] = None
    ):
        """初始化增强聊天系统"""
        self.base_url = ollama_base_url
        self.model = model
        self.memory_store = memory_store or MemoryStore()
        self.prompt_manager = prompt_manager or PromptManager()

    def _prepare_context(
        self,
        conversation_id: str,
        query: str,
        user_preferences: Dict = None
    ) -> str:
        """准备聊天上下文"""
        # 获取相关记忆
        relevant_memories = self.memory_store.retrieve_relevant_memories(
            query=query,
            conversation_id=conversation_id
        )

        # 获取默认提示词模板
        base_prompt = self.prompt_manager.get_prompt("default_chat")
        
        # 格式化记忆
        memories_text = "\n".join([
            f"- {mem['content']} (相关度: {mem['relevance_score']:.2f})"
            for mem in relevant_memories
        ])

        # 替换变量
        return self.prompt_manager.combine_prompts(
            prompt_names=["default_chat"],
            variables={
                "context": f"当前对话ID: {conversation_id}",
                "user_preferences": json.dumps(user_preferences or {}, ensure_ascii=False, indent=2),
                "memories": memories_text
            }
        )

    def chat(
        self,
        conversation_id: str,
        message: str,
        user_preferences: Dict = None,
        stream: bool = False
    ) -> Dict:
        """增强的聊天功能"""
        # 准备上下文
        context = self._prepare_context(
            conversation_id=conversation_id,
            query=message,
            user_preferences=user_preferences
        )

        # 构建请求
        data = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": context},
                {"role": "user", "content": message}
            ],
            "stream": stream
        }

        # 发送请求到Ollama
        response = requests.post(
            f"{self.base_url}/api/chat",
            json=data
        )
        response.raise_for_status()
        
        if stream:
            return response.iter_lines()
        
        result = response.json()

        # 存储对话记忆
        self.memory_store.store_memory(
            conversation_id=conversation_id,
            content=f"用户: {message}\n助手: {result['message']['content']}",
            metadata={
                "type": "dialogue",
                "user_message": message,
                "assistant_message": result['message']['content']
            }
        )

        return result

    def get_conversation_history(
        self,
        conversation_id: str,
        limit: int = 10
    ) -> List[Dict]:
        """获取对话历史"""
        return self.memory_store.retrieve_relevant_memories(
            query="",
            conversation_id=conversation_id,
            limit=limit
        )
