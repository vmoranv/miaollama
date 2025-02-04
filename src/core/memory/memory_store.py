import chromadb
from datetime import datetime
from typing import List, Dict, Optional
import json

class MemoryStore:
    def __init__(self, persist_directory: str = "./data/memories"):
        """初始化记忆存储系统"""
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name="chat_memories",
            metadata={"hnsw:space": "cosine"}
        )

    def store_memory(
        self,
        conversation_id: str,
        content: str,
        metadata: Optional[Dict] = None
    ) -> str:
        """存储新的记忆"""
        if metadata is None:
            metadata = {}
        
        # 添加时间戳
        metadata["timestamp"] = datetime.now().isoformat()
        metadata["conversation_id"] = conversation_id

        # 存储记忆
        memory_id = f"mem_{conversation_id}_{datetime.now().timestamp()}"
        self.collection.add(
            documents=[content],
            metadatas=[metadata],
            ids=[memory_id]
        )
        return memory_id

    def retrieve_relevant_memories(
        self,
        query: str,
        conversation_id: Optional[str] = None,
        limit: int = 5
    ) -> List[Dict]:
        """检索相关记忆"""
        # 构建查询条件
        where = {}
        if conversation_id:
            where["conversation_id"] = conversation_id

        # 执行相似性搜索
        results = self.collection.query(
            query_texts=[query],
            n_results=limit,
            where=where
        )

        # 格式化结果
        memories = []
        for idx, (doc, metadata) in enumerate(zip(results["documents"][0], results["metadatas"][0])):
            memories.append({
                "content": doc,
                "metadata": metadata,
                "relevance_score": results["distances"][0][idx]
            })

        return memories

    def clear_old_memories(self, days: int = 30):
        """清理旧记忆"""
        cutoff_date = datetime.now().timestamp() - (days * 24 * 60 * 60)
        # 实现记忆清理逻辑

    def export_memories(self, conversation_id: str) -> str:
        """导出特定对话的所有记忆"""
        results = self.collection.get(
            where={"conversation_id": conversation_id}
        )
        return json.dumps(results, ensure_ascii=False, indent=2)

    def import_memories(self, memories_json: str):
        """导入记忆"""
        memories = json.loads(memories_json)
        # 实现记忆导入逻辑
