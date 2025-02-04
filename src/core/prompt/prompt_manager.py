import yaml
from typing import Dict, List, Optional
import os
from datetime import datetime
import json

class PromptManager:
    def __init__(self, prompts_directory: str = "./prompts"):
        """初始化提示词管理器"""
        self.prompts_directory = prompts_directory
        self.prompts_cache = {}
        self.load_all_prompts()

    def load_all_prompts(self):
        """加载所有提示词模板"""
        if not os.path.exists(self.prompts_directory):
            os.makedirs(self.prompts_directory)
            return

        for filename in os.listdir(self.prompts_directory):
            if filename.endswith(('.yml', '.yaml')):
                prompt_name = os.path.splitext(filename)[0]
                file_path = os.path.join(self.prompts_directory, filename)
                with open(file_path, 'r', encoding='utf-8') as f:
                    self.prompts_cache[prompt_name] = yaml.safe_load(f)

    def get_prompt(self, name: str) -> Optional[Dict]:
        """获取指定提示词模板"""
        return self.prompts_cache.get(name)

    def add_prompt(self, name: str, content: Dict):
        """添加新的提示词模板"""
        file_path = os.path.join(self.prompts_directory, f"{name}.yml")
        
        # 添加元数据
        content['metadata'] = {
            'created_at': datetime.now().isoformat(),
            'version': '1.0'
        }

        # 保存到文件
        with open(file_path, 'w', encoding='utf-8') as f:
            yaml.dump(content, f, allow_unicode=True)

        # 更新缓存
        self.prompts_cache[name] = content

    def update_prompt(self, name: str, content: Dict) -> bool:
        """更新现有提示词模板"""
        if name not in self.prompts_cache:
            return False

        # 保持原有元数据
        original_metadata = self.prompts_cache[name].get('metadata', {})
        content['metadata'] = {
            **original_metadata,
            'updated_at': datetime.now().isoformat(),
            'version': str(float(original_metadata.get('version', '1.0')) + 0.1)
        }

        # 保存更新
        file_path = os.path.join(self.prompts_directory, f"{name}.yml")
        with open(file_path, 'w', encoding='utf-8') as f:
            yaml.dump(content, f, allow_unicode=True)

        # 更新缓存
        self.prompts_cache[name] = content
        return True

    def delete_prompt(self, name: str) -> bool:
        """删除提示词模板"""
        if name not in self.prompts_cache:
            return False

        file_path = os.path.join(self.prompts_directory, f"{name}.yml")
        if os.path.exists(file_path):
            os.remove(file_path)
            del self.prompts_cache[name]
            return True
        return False

    def combine_prompts(self, prompt_names: List[str], variables: Dict = None) -> str:
        """组合多个提示词模板"""
        combined_prompt = []
        
        for name in prompt_names:
            prompt = self.get_prompt(name)
            if prompt and 'content' in prompt:
                content = prompt['content']
                if variables:
                    # 替换变量
                    for key, value in variables.items():
                        content = content.replace(f"{{{key}}}", str(value))
                combined_prompt.append(content)

        return "\n".join(combined_prompt)

    def list_prompts(self) -> List[Dict]:
        """列出所有可用的提示词模板"""
        return [
            {
                'name': name,
                'metadata': prompt.get('metadata', {}),
                'description': prompt.get('description', '')
            }
            for name, prompt in self.prompts_cache.items()
        ]
