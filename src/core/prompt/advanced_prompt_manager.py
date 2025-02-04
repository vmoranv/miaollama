from typing import Dict, List, Optional
import yaml
import json
import hashlib
from datetime import datetime
from pathlib import Path
import asyncio
from ..llm.ollama_client import OllamaClient
import os
import requests

class PromptTemplate:
    def __init__(self, name: str, content: str, description: str, category: str, 
                 tags: List[str], variables: List[str], author: str, version: str,
                 created_at: str, updated_at: str, metadata: Dict = None):
        self.name = name
        self.content = content
        self.description = description
        self.category = category
        self.tags = tags
        self.variables = variables
        self.author = author
        self.version = version
        self.created_at = created_at
        self.updated_at = updated_at
        self.metadata = metadata or {}

    def dict(self):
        return {
            "name": self.name,
            "content": self.content,
            "description": self.description,
            "category": self.category,
            "tags": self.tags,
            "variables": self.variables,
            "author": self.author,
            "version": self.version,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": self.metadata
        }

class AdvancedPromptManager:
    def __init__(self):
        """初始化提示词管理器"""
        self.prompts_directory = Path(__file__).parent.parent.parent.parent / "prompts"
        self.registry_file = self.prompts_directory / "registry.json"
        
        # 确保目录存在
        self.prompts_directory.mkdir(parents=True, exist_ok=True)
        
        # 加载注册表
        if self.registry_file.exists():
            with open(self.registry_file, 'r', encoding='utf-8') as f:
                self.registry = json.load(f)
        else:
            self.registry = {
                "templates": {},
                "categories": {},
                "tags": []
            }
            self.save_registry()

    def get_template_path(self, relative_path: str) -> Path:
        """获取模板文件的完整路径"""
        # 移除路径中的 'prompts\\' 前缀（如果存在）
        cleaned_path = relative_path.replace('prompts\\', '').replace('prompts/', '')
        return self.prompts_directory / cleaned_path

    def load_registry(self):
        """加载或创建模板注册表"""
        if self.registry_file.exists():
            with open(self.registry_file, 'r', encoding='utf-8') as f:
                self.registry = json.load(f)
                # 确保tags是列表类型
                if 'tags' not in self.registry:
                    self.registry['tags'] = []
        else:
            self.registry = {
                "templates": {},
                "categories": {},
                "tags": []  # 改为列表而不是集合
            }
            self.save_registry()

    def save_registry(self):
        """保存模板注册表"""
        with open(self.registry_file, 'w', encoding='utf-8') as f:
            json.dump(self.registry, f, ensure_ascii=False, indent=2)

    def add_template(self, template: PromptTemplate) -> str:
        """添加提示词模板"""
        # 生成唯一ID
        template_id = hashlib.md5(
            f"{template.name}_{datetime.now().isoformat()}".encode()
        ).hexdigest()
        
        # 准备YAML内容
        yaml_content = {
            "name": template.name,
            "description": template.description,
            "version": template.version,
            "category": template.category,
            "tags": template.tags,
            "author": template.author,
            "created_at": template.created_at,
            "updated_at": template.updated_at,
            "variables": template.variables,
            "content": template.content
        }
        
        # 保存YAML文件
        template_path = self.prompts_directory / f"{template_id}.yml"
        with open(template_path, 'w', encoding='utf-8') as f:
            yaml.dump(yaml_content, f, allow_unicode=True, sort_keys=False, indent=2)
        
        # 更新注册表
        self.registry["templates"][template_id] = {
            "name": template.name,
            "category": template.category,
            "tags": template.tags,
            "version": template.version,
            "path": str(template_path)
        }
        
        # 更新分类
        if template.category not in self.registry["categories"]:
            self.registry["categories"][template.category] = []
        if template_id not in self.registry["categories"][template.category]:
            self.registry["categories"][template.category].append(template_id)
        
        # 更新标签
        for tag in template.tags:
            if tag not in self.registry["tags"]:
                self.registry["tags"].append(tag)
        
        # 保存注册表
        self.save_registry()
        
        return template_id

    def optimize_prompt(self, prompt: str, model: str = "llama2") -> Dict:
        """使用本地模型优化提示词"""
        try:
            # 构造优化提示词的系统提示词
            system_prompt = """你是一个提示词优化器。请严格按照以下JSON格式返回结果：

{
    "分析": "这个提示词需要优化的地方",
    "优化建议": "建议如何改进",
    "优化后的提示词": "完整的提示词文本"
}

注意：
1. 只返回上述JSON格式，不要有任何其他内容
2. 字段名称必须完全一致：分析、优化建议、优化后的提示词
3. 优化后的提示词必须是完整的文本，不要分段或添加标记
4. 不要思考，不要解释，直接返回JSON
5. 不要使用markdown格式
6. 不要包含任何换行符，所有文本都应该在一行内"""

            # 创建 OllamaClient 实例
            client = OllamaClient()
            
            # 构造对话消息
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"只返回JSON：{prompt}"}
            ]
            
            # 调用模型进行优化
            response = client.chat(messages=messages, model=model)
            
            if isinstance(response, dict):
                if "error" in response:
                    return {"error": response["error"]}
                
                content = response.get("content", "")
                print("=" * 50)
                print("原始响应内容：")
                print(content)
                print("\n字符编码信息：")
                print(f"长度：{len(content)}")
                print("特殊字符位置：")
                for i, c in enumerate(content):
                    if ord(c) > 127 or c in {'{', '}', '[', ']', '\\', '\n', '\r', '\t'}:
                        print(f"位置 {i}: {c!r} (ord={ord(c)})")
                print("=" * 50)
                
                # 尝试清理内容
                try:
                    # 移除可能的 markdown 代码块标记
                    if content.startswith('```json'):
                        content = content[7:]
                    if content.endswith('```'):
                        content = content[:-3]
                    
                    # 清理内容
                    content = content.strip()
                    print("\n清理后的内容：")
                    print(content)
                    
                    # 提取 JSON 结果
                    result = client.extract_json(content)
                    print("\n提取的 JSON：")
                    print(result)
                    
                    if result:
                        # 检查字段名是否正确
                        expected_fields = {"分析", "优化建议", "优化后的提示词"}
                        actual_fields = set(result.keys())
                        print("\n实际字段：", actual_fields)
                        
                        # 如果有字段不匹配，尝试映射
                        if actual_fields != expected_fields:
                            field_mapping = {
                                "提示词内容": "优化后的提示词",
                                "优化要点": "优化建议"
                            }
                            
                            mapped_result = {}
                            for key, value in result.items():
                                mapped_key = field_mapping.get(key, key)
                                mapped_result[mapped_key] = value
                            
                            result = mapped_result
                            print("\n映射后的结果：")
                            print(result)
                        
                        if all(k in result for k in expected_fields):
                            return result
                except Exception as e:
                    print(f"\n处理过程中的错误：{str(e)}")
                    import traceback
                    print(traceback.format_exc())
                
                return {
                    "error": "无法解析模型返回的结果",
                    "raw_content": content
                }
            
            return {"error": "模型返回格式错误"}
        except Exception as e:
            import traceback
            print(f"错误详情：{traceback.format_exc()}")
            return {"error": str(e)}

    def get_template(self, template_id: str) -> Optional[PromptTemplate]:
        """获取提示词模板"""
        if template_id not in self.registry["templates"]:
            return None

        template_path = self.registry["templates"][template_id]["path"]
        with open(template_path, 'r', encoding='utf-8') as f:
            return PromptTemplate(**yaml.safe_load(f))

    def list_templates(
        self,
        category: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> List[Dict]:
        """列出所有模板"""
        templates = []
        
        for template_id, info in self.registry["templates"].items():
            # 如果指定了分类，检查是否匹配
            if category and info["category"] != category:
                continue
                
            # 如果指定了标签，检查是否包含所有标签
            if tags and not all(tag in info["tags"] for tag in tags):
                continue
                
            try:
                template_path = Path(info["path"])
                if template_path.exists():
                    with open(template_path, 'r', encoding='utf-8') as f:
                        template_data = yaml.safe_load(f)
                        templates.append({
                            "id": template_id,
                            "name": template_data["name"],
                            "description": template_data["description"],
                            "category": template_data["category"],
                            "tags": template_data["tags"],
                            "content": template_data["content"],
                            "version": template_data["version"]
                        })
            except Exception as e:
                print(f"Error loading template {template_id}: {str(e)}")
                continue
        
        return templates

    def import_from_hub(self, hub_url: str) -> List[str]:
        """从提示词中心导入模板"""
        try:
            response = requests.get(hub_url)
            response.raise_for_status()
            templates = response.json()
            
            imported_ids = []
            for template_data in templates:
                template = PromptTemplate(**template_data)
                template_id = self.add_template(template)
                imported_ids.append(template_id)
            
            return imported_ids
        except Exception as e:
            raise Exception(f"导入失败：{str(e)}")

    def export_to_ollama(self, template_id: str, model_name: str = "llama2") -> bool:
        """导出提示词到Ollama"""
        try:
            template = self.get_template_by_id(template_id)
            if not template:
                return False
            
            # 构建Ollama请求
            response = requests.post(
                "http://localhost:11434/api/chat",
                json={
                    "model": model_name,
                    "messages": [
                        {"role": "system", "content": template["content"]}
                    ],
                    "stream": False
                }
            )
            
            return response.status_code == 200
        except Exception as e:
            print(f"Error exporting to Ollama: {str(e)}")
            return False

    def get_template_by_name(self, name: str) -> Optional[Dict]:
        """通过名称获取模板"""
        for template_id, template in self.registry["templates"].items():
            if template["name"] == name:
                template_path = self.get_template_path(template["path"])
                try:
                    with open(template_path, 'r', encoding='utf-8') as f:
                        return yaml.safe_load(f)
                except Exception as e:
                    print(f"读取模板文件失败：{str(e)}")
                    return None
        return None

    def get_template_by_id(self, template_id: str) -> Optional[Dict]:
        """通过ID获取模板"""
        if template_id in self.registry["templates"]:
            template_path = self.get_template_path(self.registry["templates"][template_id]["path"])
            try:
                with open(template_path, 'r', encoding='utf-8') as f:
                    return yaml.safe_load(f)
            except Exception as e:
                print(f"读取模板文件失败：{str(e)}")
                return None
        return None

    def update_template(self, template_id: str, template_data: Dict) -> bool:
        """更新模板"""
        if template_id in self.registry["templates"]:
            template_path = self.get_template_path(self.registry["templates"][template_id]["path"])
            try:
                # 更新文件
                with open(template_path, 'w', encoding='utf-8') as f:
                    yaml.dump(template_data, f, allow_unicode=True, sort_keys=False, indent=2)
                
                # 更新注册表
                self.registry["templates"][template_id].update({
                    "name": template_data["name"],
                    "category": template_data["category"],
                    "tags": template_data["tags"],
                    "version": template_data["version"]
                })
                
                # 更新分类
                for cat, templates in self.registry["categories"].items():
                    if template_id in templates and cat != template_data["category"]:
                        templates.remove(template_id)
                if template_data["category"] not in self.registry["categories"]:
                    self.registry["categories"][template_data["category"]] = []
                if template_id not in self.registry["categories"][template_data["category"]]:
                    self.registry["categories"][template_data["category"]].append(template_id)
                
                # 更新标签
                for tag in template_data["tags"]:
                    if tag not in self.registry["tags"]:
                        self.registry["tags"].append(tag)
                
                self.save_registry()
                return True
            except Exception as e:
                print(f"更新模板失败：{str(e)}")
                return False
        return False

    def export_template(self, template_id: str, export_path: str) -> bool:
        """导出模板到指定路径"""
        template = self.get_template_by_id(template_id)
        if template:
            try:
                with open(export_path, 'w', encoding='utf-8') as f:
                    yaml.dump(template, f, allow_unicode=True, sort_keys=False, indent=2)
                return True
            except Exception as e:
                print(f"导出模板失败：{str(e)}")
                return False
        return False

    def get_local_models(self) -> List[str]:
        """获取本地安装的Ollama模型列表"""
        try:
            response = requests.get("http://localhost:11434/api/tags")
            if response.status_code == 200:
                models = response.json()
                return [model["name"] for model in models.get("models", [])]
            return ["llama2"]  # 默认返回llama2
        except Exception:
            return ["llama2"]  # 出错时返回默认模型

    def delete_template(self, template_id: str) -> bool:
        """删除模板"""
        if template_id in self.registry["templates"]:
            try:
                # 删除文件
                template_path = self.get_template_path(self.registry["templates"][template_id]["path"])
                if template_path.exists():
                    template_path.unlink()
                
                # 从注册表中删除
                template = self.registry["templates"].pop(template_id)
                
                # 从分类中删除
                if template["category"] in self.registry["categories"]:
                    if template_id in self.registry["categories"][template["category"]]:
                        self.registry["categories"][template["category"]].remove(template_id)
                    # 如果分类为空，删除分类
                    if not self.registry["categories"][template["category"]]:
                        del self.registry["categories"][template["category"]]
                
                # 保存注册表
                self.save_registry()
                return True
            except Exception as e:
                print(f"删除模板失败：{str(e)}")
                return False
        return False

    def save_template(self, template_data: Dict) -> Optional[str]:
        """保存模板"""
        try:
            # 创建 PromptTemplate 实例
            template = PromptTemplate(
                name=template_data["name"],
                content=template_data["content"],
                description=template_data.get("description", ""),
                category=template_data.get("category", "默认分类"),
                tags=template_data.get("tags", []),
                variables=template_data.get("variables", []),
                author=template_data.get("author", "用户"),
                version=template_data.get("version", "1.0"),
                created_at=template_data.get("created_at", datetime.now().isoformat()),
                updated_at=template_data.get("updated_at", datetime.now().isoformat()),
                metadata=template_data.get("metadata", {})
            )
            
            # 添加模板
            return self.add_template(template)
        except Exception as e:
            print(f"保存模板失败：{str(e)}")
            return None
