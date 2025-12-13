import os
from typing import List, Dict, Any # 引入Any以更好地支持元数据类型
import random
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
    def get_all_courses(self) -> List[str]:
        """获取数据库中所有唯一的课程名称"""
        try:
            # 只获取 metadata，开销较小
            result = self.collection.get(include=['metadatas'])
            courses = set()
            for meta in result['metadatas']:
                if meta and 'course_name' in meta:
                    courses.add(meta['course_name'])
            return list(courses)
        except Exception as e:
            print(f"获取课程列表失败: {e}")
            return []

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
    def search(self, query: str, top_k: int = TOP_K, course_filter: str = None) -> List[Dict]:
        query_embedding = self.get_embedding(query)
        if not query_embedding: return []

        # 构造过滤条件
        where_clause = {}
        # 如果指定了课程，且不是"全局搜索"（这是一个约定俗成的保留字），则过滤
        if course_filter and course_filter not in ["全局搜索", "All"]:
            where_clause = {"course_name": course_filter}

        # 如果 where_clause 为空，ChromaDB 会忽略它（即全局搜索）
        # 注意：ChromaDB 要求 where 参数如果不传则不能为 None 或空字典，需根据版本处理
        # 最稳妥的写法是判断一下：
        kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
            "include": ['documents', 'metadatas', 'distances']
        }
        if where_clause:
            kwargs["where"] = where_clause

        results = self.collection.query(**kwargs)
        
        formatted_results = []
        if results and results.get("documents") and results.get("metadatas"):
            documents = results["documents"][0]
            metadatas = results["metadatas"][0]
            distances = results["distances"][0]
            
            for doc, meta, dist in zip(documents, metadatas, distances):
                formatted_results.append({
                    "content": doc,
                    "metadata": meta,
                    "distance": dist,
                })
        return formatted_results
    def search_hybrid(self, query: str, top_k: int = 3, pool_size: int = 20, course_filter: str = None) -> List[Dict]:
        query_embedding = self.get_embedding(query)
        if not query_embedding: return []

        # 构造过滤条件
        where_clause = {}
        if course_filter and course_filter not in ["全局搜索", "All"]:
            where_clause = {"course_name": course_filter}

        kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": pool_size,
            "include": ['documents', 'metadatas', 'distances']
        }
        if where_clause:
            kwargs["where"] = where_clause

        results = self.collection.query(**kwargs)
        
        if not results['documents']: return []

        candidates = []
        docs = results['documents'][0]
        metas = results['metadatas'][0]
        dists = results['distances'][0]

        for doc, meta, dist in zip(docs, metas, dists):
            candidates.append({"content": doc, "metadata": meta, "distance": dist})

        if not candidates: return []

        final_selection = []
        # 锚点
        final_selection.append(candidates[0])
        # 探索
        remaining = candidates[1:]
        sample_count = min(len(remaining), top_k - 1)
        if sample_count > 0:
            final_selection.extend(random.sample(remaining, sample_count))
            
        return final_selection
    def is_file_processed(self, filename: str) -> bool:
        """检查某个文件是否已经存在于向量库中"""
        # 通过 metadata 过滤查找
        result = self.collection.get(
            where={"filename": filename},
            limit=1 # 只要找到1条记录，就说明存在
        )
        return len(result['ids']) > 0


    def add_documents(self, chunks: List[Dict[str, Any]]) -> None:
        """添加文档块到向量数据库
        
        **已根据上游切分代码的输出格式进行适配和修正，解决了metadatas为空的ValueError问题。**

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
            
            # --- START: 修复和适配切分代码的输出结构 ---
            # 从 chunk 字典中提取所有非 'content' 键作为元数据
            # 您的 chunk 格式中，所有信息（如 filename, filetype, page_number等）
            # 都在顶层，这里将它们全部提取为元数据。
            metadata = {k: v for k, v in chunk.items() if k != "content"}
            
            # **关键修复：确保 metadata 字典非空，避免 ChromaDB 的 ValueError**
            # 如果元数据字典是空的，为其添加一个虚拟键，以通过 ChromaDB 的校验
            if not metadata:
                 metadata = {"_source": "virtual"}
            
            # 生成唯一的ID。使用文件的路径和块ID是最佳实践。
            # 这里对文件路径进行清理，确保生成的ID是合法的字符串。
            file_path = str(metadata.get("filepath", "no_path")).replace(os.sep, "_").replace("/", "_").replace("\\", "_")
            unique_id = f"{file_path}_{metadata.get('chunk_id', i)}_{i}" # 额外加上迭代i确保极端情况下唯一性
            # --- END: 修复和适配切分代码的输出结构 ---
            
            # 1. 获取embedding向量
            embedding = self.get_embedding(content)
            
            if embedding and content.strip(): # 确保内容和向量都有效
                # 2. 收集数据准备批量添加
                ids.append(unique_id)
                documents.append(content)
                metadatas.append(metadata)
                embeddings.append(embedding)
                print(f"准备添加文档块 ID: {unique_id}, 内容: {content}")

        # 3. 批量添加到ChromaDB collection
        if ids:
            # 由于我们已确保 metadatas 列表中的字典非空，故可以直接添加
            self.collection.upsert(
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