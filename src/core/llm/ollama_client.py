import json
import re
from typing import Dict, Iterator, List, Optional
import requests

class OllamaClient:
    """Ollama API客户端"""
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url.rstrip("/")
    
    def get_models(self) -> List[str]:
        """获取本地已安装的模型列表"""
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                data = response.json()
                models = []
                for model in data.get("models", []):
                    name = model.get("name", "").split(":")[0]  # 移除版本标签
                    if name and name not in models:
                        models.append(name)
                return models if models else ["llama2"]
            return ["llama2"]  # 默认返回llama2
        except Exception as e:
            print(f"获取模型列表失败: {str(e)}")
            return ["llama2"]  # 出错时返回默认模型

    def clean_response(self, text: str) -> str:
        """清理响应文本，移除特殊标签和格式"""
        # 处理波浪线（替换为特殊字符）
        text = text.replace('~~', '〜〜')  # 使用全角波浪线
        
        # 保留思考过程标签，但确保其格式正确
        def format_think(match):
            content = match.group(1).strip()
            # 将内容分行并添加样式
            formatted_content = []
            for line in content.split('\n'):
                if line.strip():
                    formatted_content.append(line.strip())
            
            # 使用预格式化文本块
            styled_content = '\n'.join(formatted_content)
            return f'''
<pre>
{styled_content}
</pre>
'''
        
        # 处理思考过程标签
        text = re.sub(r'<think>\s*(.*?)\s*</think>', format_think, text, flags=re.DOTALL)
        
        # 移除其他特殊标签（保留pre标签）
        text = re.sub(r'<(?!/?pre)[^>]+>', '', text)
        
        return text.strip()
    
    def format_markdown(self, text: str) -> str:
        """格式化Markdown文本"""
        # 处理代码块
        def format_code_block(match):
            lang = match.group(1) or ''
            code = match.group(2).strip()
            return f'```{lang}\n{code}\n```'
        
        text = re.sub(r'```(\w+)?\n?(.*?)```', format_code_block, text, flags=re.DOTALL)
        
        # 处理其他Markdown格式
        text = re.sub(r'\*\*(.*?)\*\*', r'**\1**', text)  # 粗体
        text = re.sub(r'(?<!\*)\*(?!\*)([^\n*]+)(?<!\*)\*(?!\*)', r'*\1*', text)  # 斜体，避免匹配多个星号
        
        return text.strip()
    
    def extract_json(self, text: str) -> Optional[Dict]:
        """从文本中提取JSON内容"""
        try:
            # 清理文本，保留换行符
            text = text.strip()
            
            # 首先尝试直接解析整个文本
            try:
                return json.loads(text)
            except:
                pass
            
            # 查找第一个完整的JSON对象
            start = text.find('{')
            end = text.rfind('}')
            if start != -1 and end != -1 and end > start:
                json_str = text[start:end + 1]
                try:
                    # 尝试解析找到的JSON字符串
                    return json.loads(json_str)
                except:
                    # 如果解析失败，尝试处理转义字符
                    try:
                        # 处理双重转义的情况
                        json_str = json_str.encode('utf-8').decode('unicode_escape')
                        return json.loads(json_str)
                    except:
                        # 如果还是失败，尝试移除所有换行符后再解析
                        json_str = re.sub(r'[\n\r]', '', json_str)
                        try:
                            return json.loads(json_str)
                        except:
                            pass
            
            return None
        except Exception as e:
            print(f"JSON提取失败: {str(e)}")
            return None

    def chat(self, messages: List[Dict], model: str = "llama2", **kwargs) -> Dict:
        """发送聊天请求"""
        url = f"{self.base_url}/api/generate"  # 修改为正确的API路径
        
        try:
            response = requests.post(
                url,
                json={
                    "model": model,
                    "prompt": messages[-1]["content"],  # 使用最后一条消息作为prompt
                    "system": messages[0]["content"] if messages and messages[0]["role"] == "system" else "",
                    "stream": False,
                    **kwargs
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                if "response" in result:
                    content = self.clean_response(result["response"])
                    return {"content": content}
                return {"error": "响应格式错误"}
            else:
                return {"error": f"请求失败：{response.status_code} - {response.text}"}
        except Exception as e:
            return {"error": str(e)}

    def chat_stream(self, messages: List[Dict], model: str = "llama2", **kwargs) -> Iterator[str]:
        """流式聊天"""
        url = f"{self.base_url}/api/chat"
        
        try:
            response = requests.post(
                url,
                json={
                    "model": model,
                    "messages": messages,
                    "stream": True,
                    **kwargs
                },
                stream=True
            )
            
            if response.status_code == 200:
                # 用于存储部分响应
                buffer = ""
                
                for line in response.iter_lines():
                    if line:
                        try:
                            # 解析JSON
                            chunk = json.loads(line)
                            if "message" in chunk and "content" in chunk["message"]:
                                content = chunk["message"]["content"]
                                buffer += content
                                
                                # 如果遇到换行符或缓冲区足够大，就清空缓冲区
                                if '\n' in buffer or len(buffer) > 80:
                                    cleaned_content = self.clean_response(buffer)
                                    cleaned_content = self.format_markdown(cleaned_content)
                                    if cleaned_content.strip():
                                        yield cleaned_content
                                    buffer = ""
                            
                        except json.JSONDecodeError:
                            continue
                
                # 处理最后的缓冲区内容
                if buffer:
                    cleaned_content = self.clean_response(buffer)
                    cleaned_content = self.format_markdown(cleaned_content)
                    if cleaned_content.strip():
                        yield cleaned_content
            else:
                yield f"请求失败：{response.status_code}"
        except Exception as e:
            yield str(e)
