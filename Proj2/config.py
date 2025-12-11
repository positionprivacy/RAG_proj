import os

# ================= API配置 =================
# 阿里云百炼的 API Key (sk-开头)
# 建议不要直接把 Key 提交到 Github/Canvas，可以使用 os.getenv 从环境变量读取，
# 或者提交前把这里删掉。
OPENAI_API_KEY = "sk-e3ab4e42d2ce42aa96f2452cb2176404" 

# 阿里云百炼兼容 OpenAI 的 Base URL (固定地址)
OPENAI_API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1" 

# 主对话模型：用于 RAG 回答问题 (对应 rag_agent.py)
# 阿里云上通常对应 "qwen-max" (通义千问-Max) 或 "deepseek-v3"
MODEL_NAME = "qwen3-max" 

# Embedding模型：用于向量化 (对应 vector_store.py)
# 阿里云提供了兼容的 embedding 模型，推荐使用 v3
OPENAI_EMBEDDING_MODEL = "text-embedding-v3" 

# [新增] 多模态模型：用于图片转文字 (对应 document_loader.py)
# 注意：这是老师模板里没有的，为了你的加分项，我们需要自己加这一行
VL_MODEL_NAME = "qwen3-vl-plus"


# ================= 数据目录配置 =================
# 存放 PDF, PPT, DOCX 的文件夹路径
DATA_DIR = "./data" 


# ================= 向量数据库配置 =================
# ChromaDB 持久化存储的路径
VECTOR_DB_PATH = "./vector_db" 

# 集合名称 (相当于数据库中的表名)
COLLECTION_NAME = "course_rag_collection" 


# ================= 文本处理配置 =================
# 切分块大小：中文文档建议 300-500 左右，太大会导致检索不准，太小会丢失语义
CHUNK_SIZE = 500 

# 重叠大小：保持上下文连贯性，建议设为 Chunk Size 的 10%-20%
CHUNK_OVERLAP = 50 

# LLM 生成回答时的最大 Token 限制 (防止回答太长废话)
MAX_TOKENS = 2000 


# ================= RAG配置 =================
# 每次检索最相似的几个文档块
# 建议 3-5 个，太少信息不够，太多会干扰模型
TOP_K = 3