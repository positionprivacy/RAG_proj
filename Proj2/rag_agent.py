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
            "4. 如果需要回答的内容与【课程内容】完全无关，你只需说明与其无关，然后使用通用能力进行回答。"
        )
        self.general_system_prompt = (
            "你是一位博学、友好的智能助手，同时也是这门课的助教。\n"
            "对于用户发起的闲聊或与课程无关的通用问题，请利用你的通用知识进行流畅、自然的回答。\n"
            "不需要局限于课程内容，也不需要引用来源。\n"
            "如果回答内容与课程相关，请引用来源，格式：`[文件名 - 页码]`"
        )
        self.quiz_system_prompt = (
            "你是一位经验丰富的助教，也是考试出题人。\n"
            "用户的请求可能包含'讲解'和'出题'两个部分，或者只是'出题'。\n"
            "**任务:**\n"
            "1. 基于【课程内容】，根据用户指令生成回答。\n"
            "2. 如果用户要求出题，请编写一道高质量的题目（选择或简答），如果用户要求，请附带标准答案和解析（解析可以折叠或放在最后）。如果用户要求不提供答案，请先不要把答案输出。\n"
            "3. 题目必须基于提供的上下文，不要凭空编造。但不必须是资料库中存在的原题。\n"
            "4. 引用来源(如果题目来源于知识库中或者知识库中有相关内容)。格式：`[文件名 - 页码]`\n"
            "5. 如果需要回答的内容与【课程内容】完全无关，你只需说明与其无关，然后使用通用能力进行回答。"
        )
        
    def _get_all_courses(self) -> List[str]:
        """
        获取所有课程名称的列表
        """
        # 从向量库中获取所有summary，提取课程名称
        all_summaries = self.vector_store.get_overall_description()
        
        # 改进的prompt，明确要求JSON格式
        prompt = f"""
    请从以下课程摘要中提取所有课程名称/三级学科名称。

    **要求：**
    1. 只提取课程名称，不要其他内容
    2. 返回严格的JSON数组格式
    3. 格式示例：["课程1", "课程2", "课程3"]
    4. 不要包含任何解释性文字

    课程摘要：
    {all_summaries}

    请返回JSON数组：
    """
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "你是一个JSON生成器，只返回有效的JSON数组，不要任何额外文字。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,  # 更低的温度保证格式
            max_tokens=500
        )

        content = response.choices[0].message.content.strip()

        course_names = json.loads(content)
        return course_names

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
    def _manage_history(self, history: List[Dict], max_rounds: int = 10) -> List[Dict]:
        """
        [Feature 4] 历史记录管理：滑动窗口
        保留 System Prompt，并截取最近的 max_rounds 轮对话。
        """
        if not history:
            return []
        
        # 如果历史记录很少，直接返回
        # 假设一轮包含 user 和 assistant 两条，所以乘以 2
        if len(history) <= max_rounds * 2:
            return history
            
        # 策略：保留最早的 System Prompt (如果有)，然后取最新的 N 条
        # 这样既能记住人设，又能记住最近的上下文
        managed_history = []
        
        # 检查第一条是不是 system，如果是，保留它
        start_idx = 0
        if history[0]["role"] == "system":
            managed_history.append(history[0])
            start_idx = 1
            
        # 截取最后 N 条 (倒数)
        # 例如保留最近 20 条消息 (10轮)
        keep_count = max_rounds * 2
        recent_history = history[-keep_count:]
        
        managed_history.extend(recent_history)
        
        print(f"  [History] 已截断历史记录: {len(history)} -> {len(managed_history)} 条")
        return managed_history

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
        system_prompt: str = "",
        temperature: float = 0.7
    ) -> str:
        """
        生成回答：根据是否有 Context 动态构建 User Prompt
        """
        clean_sys_prompt = self._clean_text(system_prompt)
        messages = [{"role": "system", "content": system_prompt}]
        if chat_history:
            # 清洗历史
            cleaned_history = []
            for msg in chat_history:
                cleaned_history.append({
                    "role": msg["role"],
                    "content": self._clean_text(msg["content"])
                })
            
            # 截断历史 (保留最近 10 轮)
            short_history = self._manage_history(cleaned_history, max_rounds=10)
            messages.extend(short_history)

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
                model=self.model, messages=messages, temperature=temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"生成回答出错: {str(e)}"

    def answer_question(
        self, query: str, chat_history: Optional[List[Dict]] = None, top_k: int = TOP_K,course_filter: str = "全局搜索",temperature: float = 0.7
    ) -> str:
        actual_query_to_llm = query 
        # 1. 意图路由
        print("  Thinking: 分析意图...")
        intent = self._analyze_intent(query)
        intent_type = intent.get("type", "qa")
        topic = intent.get("topic", query)
        print(f"  Intent: [{intent_type}] Topic: [{topic}]")

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
            retrieved_docs = self.vector_store.search_hybrid(query=topic, top_k=top_k, pool_size=20,course_filter=course_filter)
            final_context = self._format_context(retrieved_docs)
            if not final_context: final_context = "（未找到相关资料，请尝试根据通用知识出题）"

        # 分支 C: 课程问答 -> 标准检索
        else: # qa
            print(f"  Retrieving: 搜索 [{topic}] in [{course_filter}]...")
            
            # [修改] 使用带 Rerank 的检索
            retrieved_docs = self.vector_store.search_with_rerank(
                query=topic, 
                top_k=top_k, 
                course_filter=course_filter
            )
            
            # [算法优化] 基于 Rerank 分数的智能拒识
            # GTE-Rerank 的分数通常在 0~1 之间 (Sigmoid output)
            # 经验阈值：< 0.35 通常表示不太相关，< 0.1 表示完全无关
            # 你可以根据实际测试调整这个阈值
            RERANK_THRESHOLD = 0.25
            
            is_relevant = False
            
            if retrieved_docs:
                # 检查第一条（最相关的一条）的分数
                # 注意：如果降级为普通检索，可能没有 rerank_score 字段，需兼容
                top_score = retrieved_docs[0].get('rerank_score')
                
                if top_score is not None:
                    # Rerank 成功，使用精确分数判断
                    if top_score > RERANK_THRESHOLD:
                        is_relevant = True
                    else:
                        print(f"  ! Rerank 分数过低 ({top_score:.4f})，判定无关")
                else:
                    # Rerank 失败或降级，使用原来的 distance 判断 (兼容旧逻辑)
                    top_dist = retrieved_docs[0].get('distance', 999)
                    if top_dist < 1.5: # 假设向量距离阈值
                        is_relevant = True

            if is_relevant:
                # Case C1: 找到了高质量资料 -> 正常 RAG
                current_system_prompt = self.rag_system_prompt
                final_context = self._format_context(retrieved_docs)
            else:
                # Case C2: 没找到 或 结果太差 -> 优雅降级
                print("  ! 检索结果为空或不相关，切换至通用回答模式")
                
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
            system_prompt=current_system_prompt,
            temperature=temperature
        )

        return answer
    def generate_session_title(self, query: str) -> str:
        """
        [新增] 根据用户的第一句输入，利用大模型生成一个简短的会话标题
        """
        prompt = f"""
        请根据用户的输入生成一个极其简短的对话标题。
        要求：
        1. 长度控制在 6-10 个汉字以内。
        2. 不要包含“用户问”、“关于”等废话，直接概括核心主题。
        3. 不要使用引号或标点符号。
        
        用户输入："{query}"
        """
        try:
            # 使用较低的 temperature 保证标题简洁稳定
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3, 
                max_tokens=20 # 限制输出长度
            )
            title = response.choices[0].message.content.strip()
            
            # 清洗一下，防止模型输出引号
            return title.replace('"', '').replace("'", "").replace("。", "")
        except Exception as e:
            print(f"标题生成失败: {e}")
            # 降级策略：如果生成失败，还是截取前10个字
            return query[:10] + "..."

    def chat(self) -> None:
        """交互式对话"""
        
        ASSISTANT_NAME = "dinner" # 助教系统名称

        dinner_ascii = r"""
    ██████╗  ██╗ ███╗   ██╗ ███╗   ██╗ ███████╗ ██████╗ 
    ██╔══██╗ ██║ ████╗  ██║ ████╗  ██║ ██╔════╝ ██╔══██╗
    ██║  ██║ ██║ ██╔██╗ ██║ ██╔██╗ ██║ █████╗   ██████╔╝
    ██║  ██║ ██║ ██║╚██╗██║ ██║╚██╗██║ ██╔══╝   ██╔══██╗
    ██████╔╝ ██║ ██║ ╚████║ ██║ ╚████║ ███████╗ ██║  ██║
    ╚═════╝  ╚═╝ ╚═╝  ╚═══╝╚ ═╝  ╚═══╝ ╚══════╝ ╚═╝  ╚═╝
        """

        
        # 1. 打印助教系统名称和欢迎信息
        print("=" * 60)
        print(dinner_ascii)
        print(f"🌟 欢迎使用【{ASSISTANT_NAME}】智能课程助教系统！")
        print("（已启用知识库检索、意图路由及习题出题功能）")
        print("-" * 60)
        
        # 2. 获取并打印课程列表
        try:
            print("⏳ 正在加载知识库中的课程列表...")
            # 注意：self._get_all_courses() 需要调用外部 API，可能会耗时
            course_names = self._get_all_courses()
            if course_names and isinstance(course_names, list):
                print("📚 当前知识库包含的课程：")
                for i, course in enumerate(course_names):
                    print(f" {i+1}. {course}")
            else:
                print("⚠️ 未能加载课程列表或知识库为空。")
        except Exception as e:
            print(f"❌ 加载课程列表时发生错误: {e}")
            
        print("=" * 60)
        
        chat_history = []

        while True:
            
            # 改进输入提示
            query = input(f"\n👤 学生提问 : ").strip()

            if not query:
                continue

            if query.lower() in ["exit", "quit", "bye", "退出"]:
                print("👋 感谢使用课程助教系统，期待下次见面！再见！")
                break

            answer = self.answer_question(query, chat_history=chat_history)



            # 改进输出提示
            print(f"\n💡 助教: {answer}")

            # 更新对话历史 (仅存储用户查询和助教回答，用于维护上下文)
            chat_history.append({"role": "user", "content": query})
            chat_history.append({"role": "assistant", "content": answer})
