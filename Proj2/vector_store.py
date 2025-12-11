import os
from typing import List, Dict

import chromadb
from chromadb.config import Settings
from openai import OpenAI
from tqdm import tqdm

from config import (
    VECTOR_DB_PATH,
    COLLECTION_NAME,
    OPENAI_API_KEY,
    OPENAI_API_BASE,
    OPENAI_EMBEDDING_MODEL,
    TOP_K,
)


class VectorStore:

    def __init__(
        self,
        db_path: str = VECTOR_DB_PATH,
        collection_name: str = COLLECTION_NAME,
        api_key: str = OPENAI_API_KEY,
        api_base: str = OPENAI_API_BASE,
    ):
        self.db_path = db_path
        self.collection_name = collection_name

        # 初始化OpenAI客户端
        self.client = OpenAI(api_key=api_key, base_url=api_base)
        self.embedding_model = OPENAI_EMBEDDING_MODEL  # 添加模型名称属性

        # 初始化ChromaDB
        os.makedirs(db_path, exist_ok=True)
        self.chroma_client = chromadb.PersistentClient(
            path=db_path, settings=Settings(anonymized_telemetry=False)
        )

        # 获取或创建collection
        # 注意: ChromaDB在创建collection时不支持直接指定外部embedding model，
        # 我们将在代码中手动处理embedding。
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name, metadata={"description": "课程材料向量数据库"}
        )

    def get_embedding(self, text: str) -> List[float]:
        """获取文本的向量表示

        使用OpenAI API获取文本的embedding向量
        """
        # 移除文本中的换行符，避免一些模型出错或影响效果
        text = text.replace("\n", " ") 
        
        try:
            response = self.client.embeddings.create(
                input=[text],
                model=self.embedding_model # 使用配置中指定的模型
            )
            # 返回第一个（也是唯一的）embedding向量
            return response.data[0].embedding
        except Exception as e:
            print(f"获取embedding失败: {e}")
            return []


    def add_documents(self, chunks: List[Dict[str, str]]) -> None:
        """添加文档块到向量数据库
        实现文档块添加到向量数据库
        要求：
        1. 遍历文档块
        2. 获取文档块内容
        3. 获取文档块元数据
        5. 打印添加进度
        """
        
        # 批次添加的列表
        ids: List[str] = []
        documents: List[str] = []
        metadatas: List[Dict] = []
        embeddings: List[List[float]] = []

        # ChromaDB要求ID是字符串，我们使用文档块的索引作为ID
        for i, chunk in enumerate(tqdm(chunks, desc="正在生成和添加文档向量")):
            content = chunk.get("content", "")
            metadata = chunk.get("metadata", {})
            
            # 1. 获取embedding向量
            embedding = self.get_embedding(content)
            
            if embedding:
                # 2. 收集数据准备批量添加
                ids.append(f"doc_{i}")
                documents.append(content)
                metadatas.append(metadata)
                embeddings.append(embedding)

        # 3. 批量添加到ChromaDB collection
        if ids:
            self.collection.add(
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
                ids=ids,
            )
            print(f"\n成功添加 {len(ids)} 个文档块到向量数据库。")
        else:
            print("\n没有文档块被添加到向量数据库。")


    def search(self, query: str, top_k: int = TOP_K) -> List[Dict]:
        """搜索相关文档

        实现向量相似度搜索
        要求：
        1. 首先获取查询文本的embedding向量（调用self.get_embedding）
        2. 使用self.collection进行向量搜索, 得到top_k个结果
        3. 格式化返回结果，每个结果包含：
           - content: 文档内容
           - metadata: 元数据（文件名、页码等）
        4. 返回格式化的结果列表
        """

        # 1. 获取查询文本的embedding向量
        query_embedding = self.get_embedding(query)
        
        if not query_embedding:
            print("查询embedding生成失败，无法进行搜索。")
            return []

        # 2. 使用self.collection进行向量搜索
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=['documents', 'metadatas', 'distances'] # 包含文档内容、元数据和距离（相似度）
        )
        
        # 3. 格式化返回结果
        formatted_results: List[Dict] = []
        
        # ChromaDB的查询结果是嵌套列表，需要解包
        if results and results.get("documents") and results.get("metadatas"):
            
            # 假设只查询了一个embedding，所以我们取第一个元素 [0]
            documents = results["documents"][0]
            metadatas = results["metadatas"][0]
            distances = results["distances"][0]
            
            for doc, meta, dist in zip(documents, metadatas, distances):
                formatted_results.append(
                    {
                        "content": doc,
                        "metadata": meta,
                        # 距离越小，相似度越高。1 - distance 简单估算相似度
                        "distance": dist, 
                    }
                )
                
        return formatted_results

    def clear_collection(self) -> None:
        """清空collection"""
        self.chroma_client.delete_collection(name=self.collection_name)
        # 重新创建collection
        self.collection = self.chroma_client.get_or_create_collection(
            name=self.collection_name, metadata={"description": "课程向量数据库"}
        )
        print("向量数据库已清空")

    def get_collection_count(self) -> int:
        """获取collection中的文档数量"""
        return self.collection.count()