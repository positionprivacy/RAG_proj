from typing import List, Dict, Optional, Tuple, Any

from openai import OpenAI

from config import (
    OPENAI_API_KEY,
    OPENAI_API_BASE,
    MODEL_NAME,
    TOP_K,
)
from vector_store import VectorStore


class RAGAgent:
    def __init__(
        self,
        model: str = MODEL_NAME,
    ):
        self.model = model

        self.client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_API_BASE)

        self.vector_store = VectorStore()

        """
        TODO: 实现并调整系统提示词，使其符合课程助教的角色和回答策略
        """
        self.system_prompt = (
            "你是一位友好、专业且细心的课程助教，正在回答学生关于课程材料和作业的问题。\n"
            "你的回答必须严格基于提供的【课程内容】上下文。\n"
            "**回答策略:**\n"
            "1. 仔细阅读【课程内容】。\n"
            "2. 基于这些内容，用清晰、简洁的中文回答学生的问题。\n"
            "3. 如果【课程内容】中没有足够的信息来回答问题，请礼貌地告知学生：'根据我目前的课程材料，我无法找到关于这个问题确切的答案。'\n"
            "4. 务必在回答的末尾，使用 Markdown 格式（例如 `[文件名 - 页码X]`）列出所有你引用到的**来源信息**，以便学生查阅。\n"
            "5. 保持助教的专业和鼓励语气。"
        )

    def retrieve_context(
        self, query: str, top_k: int = TOP_K
    ) -> Tuple[str, List[Dict]]:
        """检索相关上下文
        实现检索相关上下文
        要求：
        1. 使用向量数据库检索相关文档
        2. 格式化检索结果，构建上下文字符串
        3. 每个检索结果需要包含来源信息（文件名和页码）
        4. 返回格式化的上下文字符串和原始检索结果列表
        """
        
        # 1. 使用向量数据库检索相关文档
        retrieved_docs: List[Dict[str, Any]] = self.vector_store.search(query=query, top_k=top_k)
        
        context_parts: List[str] = []

        # 2. 格式化检索结果，构建上下文字符串
        for i, doc in enumerate(retrieved_docs):
            content = doc.get("content", "")
            metadata = doc.get("metadata", {})
            
            # --- START: 适配上游 chunk 格式 ---
            # 上游 chunk 格式中，文件信息和页码是顶层元数据
            file_name = metadata.get("filename", "未知文件") # 使用 filename 键
            page_number = metadata.get("page_number", 0)    # 使用 page_number 键
            
            # 格式化页码：如果 page_number > 0，则显示页码；否则显示“无页码”
            page_label = f"页码{page_number}" if page_number else "无页码"
            # --- END: 适配上游 chunk 格式 ---
            
            # 3. 每个检索结果包含来源信息
            # 格式化上下文，便于LLM处理
            source_info = f"[来源: {file_name} - {page_label}]"
            context_parts.append(f"--- 课程内容片段 {i+1} ---\n{content}\n{source_info}\n")
            
        context = "\n".join(context_parts)
        
        # 4. 返回格式化的上下文字符串和原始检索结果列表
        return context, retrieved_docs

    def generate_response(
        self,
        query: str,
        context: str,
        chat_history: Optional[List[Dict]] = None,
    ) -> str:
        """生成回答
        
        参数:
            query: 用户问题
            context: 检索到的上下文
            chat_history: 对话历史
        """
        messages = [{"role": "system", "content": self.system_prompt}]

        if chat_history:
            messages.extend(chat_history)

        """
        TODO: 实现用户提示词
        要求：
        1. 包含相关的课程内容
        2. 包含学生问题
        3. 包含来源信息（文件名和页码）
        4. 返回用户提示词
        """
        # 构造用户提示词，将上下文、问题和来源信息全部打包

        print("生成回答时使用的上下文片段:", {context})
        user_text = f"""
请基于下面提供的【课程内容】来回答学生的问题。

---
【学生问题】
{query}

---
【课程内容】
{context}

---
请严格按照系统提示词的要求来组织你的回答。
"""

        messages.append({"role": "user", "content": user_text})
        
        # 多模态接口示意（如需添加图片支持，可参考以下格式）：
        # content_parts = [{"type": "text", "text": user_text}]
        # content_parts.append({
        #    "type": "image_url",
        #    "image_url": {"url": f"data:image/png;base64,{base64_image}"}
        # })
        # messages.append({"role": "user", "content": content_parts})

        try:
            response = self.client.chat.completions.create(
                model=self.model, messages=messages, temperature=0.7, max_tokens=1500
            )

            return response.choices[0].message.content
        except Exception as e:
            return f"生成回答时出错: {str(e)}"

    def answer_question(
        self, query: str, chat_history: Optional[List[Dict]] = None, top_k: int = TOP_K
    ) -> str: # 修改返回类型为 str (根据函数体)
        """回答问题
        
        参数:
            query: 用户问题
            chat_history: 对话历史
            top_k: 检索文档数量
            
        返回:
            生成的回答
        """
        # 核心RAG流程：检索 -> 增强 -> 生成
        context, retrieved_docs = self.retrieve_context(query, top_k=top_k) # 

        if not context:
            # 当未检索到任何内容时的默认上下文
            context = "（未检索到特别相关的课程材料，请告知学生）"

        answer = self.generate_response(query, context, chat_history)

        # 这里我们只返回了回答，如果您希望在最终结果中包含检索到的文档，可以修改返回类型
        return answer

    def chat(self) -> None:
        """交互式对话"""
        print("=" * 60)
        print("欢迎使用智能课程助教系统！")
        print("=" * 60)

        chat_history = []

        while True:
            try:
                query = input("\n学生: ").strip()

                if not query:
                    continue
                
                # 检查退出命令
                if query.lower() in ["quit", "exit", "退出"]:
                    print("\n助教: 感谢您的提问，再见！")
                    break

                answer = self.answer_question(query, chat_history=chat_history)

                print(f"\n助教: {answer}")

                # 更新对话历史 (仅存储用户查询和助教回答，用于维护上下文)
                chat_history.append({"role": "user", "content": query})
                chat_history.append({"role": "assistant", "content": answer})

            except Exception as e:
                print(f"\n错误: {str(e)}")