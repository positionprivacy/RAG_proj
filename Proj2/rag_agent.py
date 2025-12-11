from typing import List, Dict, Optional, Tuple, Any
import json
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
        self.rag_system_prompt = (
            "你是一位友好、专业且细心的课程助教。\n"
            "你的任务是基于提供的【课程内容】回答学生问题。\n"
            "**规则:**\n"
            "1. 仔细阅读上下文，用清晰的中文回答。\n"
            "2. 如果问题与课程无关且不属于闲聊，请礼貌拒绝。\n"
            "3. 必须在回答末尾标注来源，格式：`[文件名 - 页码]`。\n"
        )
        self.general_system_prompt = (
            "你是一位博学、友好的智能助手，同时也是这门课的助教。\n"
            "对于用户发起的闲聊或与课程无关的通用问题，请利用你的通用知识进行流畅、自然的回答。\n"
            "不需要局限于课程内容，也不需要引用来源。"
        )
        self.quiz_system_prompt = (
                "你是一位经验丰富的助教，也是考试出题人。\n"
                "用户的请求可能包含'讲解'和'出题'两个部分，或者只是'出题'。\n"
                "**任务:**\n"
                "1. 基于【课程内容】，根据用户指令生成回答。\n"
                "2. 如果用户要求出题，请编写一道高质量的题目（选择或简答），如果用户要求，请附带标准答案和解析（解析可以折叠或放在最后）。如果用户要求不提供答案，请先不要把答案输出。\n"
                "3. 题目必须基于提供的上下文，不要凭空编造。但不必须是资料库中存在的原题。\n"
                "4. 引用来源(如果题目来源于知识库中)。"
            )
    def _clean_text(self, text: str) -> str:
        """
        清洗字符串，移除无法编码的代理字符（Surrogates），防止 API 调用崩溃
        """
        if not text:
            return ""
        try:
            # 尝试编码再解码，忽略错误字符
            return text.encode('utf-8', 'ignore').decode('utf-8')
        except Exception:
            # 如果彻底失败，返回空或原始值
            return ""
    def _analyze_intent(self, query: str) -> Dict:
        """
        使用 LLM 分析用户意图，进行路由。
        返回 JSON: {type, topic}
        """
        prompt = f"""
        你是一个意图分类器。分析用户输入："{query}"
        
        返回严格的 JSON 格式（无Markdown），包含字段：
        1. type: (str) 
           - "greeting": 打招呼/闲聊 (如"你好", "在吗")
           - "irrelevant": 与课程/研究/学习完全无关 (如"今天天气", "讲个笑话")
           - "quiz": 要求出题/测验/考试/考考我 (包含"讲解并出题"的混合意图，只要有出题需求就算)
           - "qa": 普通课程提问 (默认)
        2. topic: (str) 提取核心知识点关键词，如果是闲聊则为空。不需要太精简，可以尽量保持原始内容。
        
        示例："你好" -> {{"type": "greeting", "topic": ""}}
        示例："出个Attention的题" -> {{"type": "quiz", "topic": "Attention"}}
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1, # 低温保证格式
                max_tokens=100
            )
            content = response.choices[0].message.content.strip()
            # 清洗可能存在的 Markdown 标记
            if content.startswith("```"):
                content = content.replace("```json", "").replace("```", "")
            return json.loads(content)
        except:
            # 降级：默认 QA
            return {"type": "qa", "topic": query}
    def _format_context(self, docs: List[Dict]) -> str:
        context_parts = []
        for i, doc in enumerate(docs):
            content = doc.get("content", "")
            metadata = doc.get("metadata", {})
            file_name = metadata.get("filename", "未知文件")
            page = metadata.get("page_number", 0)
            page_label = f"页码{page}" if page else "无页码"
            
            source_info = f"[来源: {file_name} - {page_label}]"
            context_parts.append(f"--- 片段 {i+1} ---\n{content}\n{source_info}\n")
        return "\n".join(context_parts)

    def retrieve_context(
        self, query: str, top_k: int = TOP_K
    ) -> Tuple[str, List[Dict]]:
        """
        标准检索方法 (用于普通问答)
        """
        # 1. 使用标准检索
        retrieved_docs = self.vector_store.search(query=query, top_k=top_k)
        
        # 2. 调用通用格式化函数
        context = self._format_context(retrieved_docs)
        
        return context, retrieved_docs

    def generate_response(
        self,
        query: str,
        context: Optional[str] = None,
        chat_history: Optional[List[Dict]] = None,
        system_prompt: str = ""
    ) -> str:
        """
        生成回答：根据是否有 Context 动态构建 User Prompt
        """
        clean_sys_prompt = self._clean_text(system_prompt)
        messages = [{"role": "system", "content": system_prompt}]
        if chat_history:
            for msg in chat_history:
                clean_msg = {
                    "role": msg["role"],
                    "content": self._clean_text(msg["content"])
                }
                messages.append(clean_msg)

        # === 动态构建 User Prompt ===
        if context:
            # 场景 A: 有 RAG 上下文 (QA / Quiz)
            clean_context = self._clean_text(context)
            clean_query = self._clean_text(query)
            
            user_text = f"""
请基于下面提供的【课程内容】来回答用户指令。

---
【用户指令】
{clean_query}

---
【课程内容】
{clean_context}

---
请严格按照系统提示词的要求来组织你的回答。
"""
        else:
            # 场景 B: 无上下文 (Greeting / Irrelevant)
            # 直接把用户问题发给大模型，不加任何 RAG 限制
            user_text = query

        messages.append({"role": "user", "content": user_text})

        try:

            print(f"投喂内容：{messages}")
            response = self.client.chat.completions.create(
                model=self.model, messages=messages, temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"生成回答出错: {str(e)}"

    def answer_question(
        self, query: str, chat_history: Optional[List[Dict]] = None, top_k: int = TOP_K
    ) -> str:
        actual_query_to_llm = query 
        # 1. 意图路由
        print("  Thinking: 分析意图...")
        intent = self._analyze_intent(query)
        intent_type = intent.get("type", "qa")
        topic = intent.get("topic", query)
        print(f"  Intent: [{intent_type}] Topic: [{topic}]")

        retrieved_docs = []
        final_context = None
        current_system_prompt = self.general_system_prompt # 默认为通用

        # === 路由分支 ===

        # 分支 A: 纯通用模式 (Greeting / Irrelevant) -> 不检索
        if intent_type in ["greeting", "irrelevant"]:
            # 不检索，final_context 保持为 None
            # System Prompt 使用 general_system_prompt
            pass 

        # 分支 B: 出题模式 -> 混合检索
        elif intent_type == "quiz":
            current_system_prompt = self.quiz_system_prompt
            retrieved_docs = self.vector_store.search_hybrid(query=topic, top_k=top_k, pool_size=20)
            final_context = self._format_context(retrieved_docs)
            if not final_context: final_context = "（未找到相关资料，请尝试根据通用知识出题）"

        # 分支 C: 课程问答 -> 标准检索
        else: # qa
            print(f"  Retrieving: 搜索 [{topic}]...")
            retrieved_docs = self.vector_store.search(query=topic, top_k=top_k)
            
            # === [新增] 阈值判断逻辑 ===
            # 假设阈值设为 1.5 (你可以根据 log 调整)
            SIMILARITY_THRESHOLD = 1.0 
            
            is_relevant = False
            
            if retrieved_docs:
                # 获取第一条结果的距离
                first_dist = retrieved_docs[0].get("distance", 999)
                print(f"  > Top-1 Distance: {first_dist:.4f}") # 打印出来方便调试
                
                if first_dist < SIMILARITY_THRESHOLD:
                    is_relevant = True
                else:
                    print(f"  ! 距离过大 (>{SIMILARITY_THRESHOLD})，判定为无关内容")
            
            # === 分支判断 ===
            
            if is_relevant:
                # Case C1: 找到了 *高质量* 资料 -> 正常 RAG
                current_system_prompt = self.rag_system_prompt
                final_context = self._format_context(retrieved_docs)
            else:
                # Case C2: 没找到 或 结果太差 -> 优雅降级
                print("  ! 检索结果为空或不相关，切换至通用回答模式")
                
                current_system_prompt = self.general_system_prompt
                final_context = None
                
                # 注入通用回答指令
                actual_query_to_llm = (
                    f"{query}\n\n"
                    "---------------------\n"
                    "【系统指令】\n"
                    "知识库检索结果为空（或相关度过低）。这意味着课程资料中没有提及此内容。\n"
                    "请执行以下操作：\n"
                    "1. 首先明确声明：'**根据当前的课程资料，未找到与该问题相关的内容。**'\n"
                    "2. 然后说：'以下是我基于通用知识为您提供的解答：'\n"
                    "3. 最后基于你的通用知识库回答该问题。"
                )

        # 2. 生成回答
        # 注意：这里传入的是 actual_query_to_llm
        answer = self.generate_response(
            query=actual_query_to_llm, 
            context=final_context, 
            chat_history=chat_history,
            system_prompt=current_system_prompt
        )

        return answer

    def chat(self) -> None:
        """交互式对话"""
        print("=" * 60)
        print("欢迎使用智能课程助教系统！(已启用意图路由 & 随机出题)")
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