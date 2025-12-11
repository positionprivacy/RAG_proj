# test_api.py
from openai import OpenAI
import config

client = OpenAI(
    api_key=config.OPENAI_API_KEY,
    base_url=config.OPENAI_API_BASE
)

# 测试对话
try:
    print("正在测试对话模型...")
    completion = client.chat.completions.create(
        model=config.MODEL_NAME,
        messages=[{"role": "user", "content": "你好，测试一下"}]
    )
    print("对话成功:", completion.choices[0].message.content)
except Exception as e:
    print("对话配置有误:", e)

# 测试 Embedding
try:
    print("\n正在测试 Embedding...")
    emb = client.embeddings.create(
        model=config.OPENAI_EMBEDDING_MODEL,
        input="测试文本"
    )
    print("Embedding 成功，维度:", len(emb.data[0].embedding))
except Exception as e:
    print("Embedding 配置有误:", e)