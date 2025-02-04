from typing import Dict, List, Optional, Generator, Union
from .ollama_client import OllamaClient
import json
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

class EnhancedOllama:
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama2",
        context_window: int = 4096,
        max_tokens: int = 2048
    ):
        self.client = OllamaClient(base_url, model)
        self.context_window = context_window
        self.max_tokens = max_tokens
        self.context = []
        self._executor = ThreadPoolExecutor(max_workers=4)

    def set_model_params(
        self,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        repeat_penalty: float = 1.1,
        **kwargs
    ) -> Dict:
        """设置模型参数"""
        self.model_params = {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repeat_penalty": repeat_penalty,
            **kwargs
        }
        return self.model_params

    def _prepare_context(self, messages: List[Dict]) -> List[Dict]:
        """准备上下文"""
        # 合并历史上下文和新消息
        all_messages = self.context + messages
        
        # 计算token数（这里使用简单的字符计数作为估算）
        total_length = sum(len(str(m)) for m in all_messages)
        
        # 如果超出上下文窗口，裁剪旧消息
        while total_length > self.context_window and len(all_messages) > 1:
            removed = all_messages.pop(0)
            total_length -= len(str(removed))
        
        return all_messages

    def chat(
        self,
        messages: List[Dict],
        system_prompt: Optional[str] = None,
        stream: bool = False,
        remember_context: bool = True,
        **kwargs
    ) -> Union[Dict, Generator]:
        """增强的聊天功能"""
        # 准备上下文
        prepared_messages = self._prepare_context(messages)
        
        # 添加系统提示词
        if system_prompt:
            prepared_messages.insert(0, {
                "role": "system",
                "content": system_prompt
            })
        
        # 合并模型参数
        params = {
            **(self.model_params if hasattr(self, 'model_params') else {}),
            **kwargs
        }
        
        # 发送请求
        response = self.client.chat(
            messages=prepared_messages,
            stream=stream,
            **params
        )
        
        # 更新上下文
        if remember_context and not stream:
            self.context = prepared_messages + [{
                "role": "assistant",
                "content": response['message']['content']
            }]
        
        return response

    async def batch_process(
        self,
        prompts: List[str],
        system_prompt: Optional[str] = None,
        max_concurrent: int = 3
    ) -> List[Dict]:
        """批量处理多个提示词"""
        async def process_single(prompt: str) -> Dict:
            async with aiohttp.ClientSession() as session:
                url = f"{self.client.base_url}/api/chat"
                data = {
                    "model": self.client.model,
                    "messages": [
                        *(
                            [{"role": "system", "content": system_prompt}]
                            if system_prompt else []
                        ),
                        {"role": "user", "content": prompt}
                    ],
                    **(self.model_params if hasattr(self, 'model_params') else {})
                }
                
                async with session.post(url, json=data) as response:
                    return await response.json()

        # 使用信号量限制并发数
        sem = asyncio.Semaphore(max_concurrent)
        
        async def bounded_process(prompt: str) -> Dict:
            async with sem:
                return await process_single(prompt)

        # 并发处理所有提示词
        tasks = [bounded_process(prompt) for prompt in prompts]
        return await asyncio.gather(*tasks)

    def format_prompt(
        self,
        template: str,
        variables: Dict,
        system_prompt: Optional[str] = None
    ) -> str:
        """格式化提示词模板"""
        # 替换变量
        formatted = template
        for key, value in variables.items():
            formatted = formatted.replace(f"{{{key}}}", str(value))
        
        # 添加系统提示词
        if system_prompt:
            formatted = f"{system_prompt}\n\n{formatted}"
        
        return formatted

    def clear_context(self):
        """清除上下文历史"""
        self.context = []

    def get_model_info(self) -> Dict:
        """获取模型信息"""
        try:
            models = self.client.list_models()
            for model in models.get('models', []):
                if model['name'] == self.client.model:
                    return model
        except Exception as e:
            return {"error": str(e)}
        return {"error": "Model not found"}
