import streamlit as st
import os
import shutil
from rag_agent import RAGAgent
from config import MODEL_NAME, DATA_DIR
# 引入刚才封装的处理函数
from process_data import process_single_file 

# 设置页面配置
st.set_page_config(page_title="智能课程助教", page_icon="🎓", layout="wide")

# 初始化 Agent
@st.cache_resource
def init_agent():
    return RAGAgent(model=MODEL_NAME)

if "agent" not in st.session_state:
    st.session_state.agent = init_agent()

agent = st.session_state.agent

# ===================== 侧边栏 =====================
with st.sidebar:
    st.title("🎓 助教控制台")
    
    # 使用 Tabs 分离"对话设置"和"知识库管理"
    tab_chat, tab_kb = st.tabs(["💬 对话设置", "📂 知识库管理"])
    
    # --- Tab 1: 对话设置 ---
    with tab_chat:
        st.subheader("🔍 检索与模型设置")
        
        # 1. 课程分区选择 (保持不变)
        available_courses = agent.vector_store.get_all_courses()
        options = ["全局搜索"] + available_courses
        selected_course = st.selectbox("选择检索分区:", options, index=0)
        
        st.divider()
        
        # 2. [新增] Top-K 滑块
        st.write("###### ⚙️ 检索参数")
        selected_top_k = st.slider(
            "检索数量 (Top-K):",
            min_value=1,
            max_value=10,
            value=3,
            help="每次回答参考的资料片段数量。数量越多信息越全，但可能引入噪声。"
        )
        
        # 3. [新增] Temperature 滑块
        st.write("###### 🧠 模型参数")
        selected_temperature = st.slider(
            "创造力 (Temperature):",
            min_value=0.0,
            max_value=1.5,
            value=0.7,
            step=0.1,
            help="数值越高，回答越随机、有创造力（适合出题）；数值越低，回答越严谨、稳定（适合定义解释）。"
        )
        
        st.caption(f"当前配置: Top-{selected_top_k} | Temp-{selected_temperature}")
        
        if st.button("🗑️ 清空对话", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    # --- Tab 2: 知识库管理 (核心修改) ---
    # --- Tab 2: 知识库管理 ---
    with tab_kb:
        st.subheader("📤 批量上传文件")
        
        # [修改 1] 开启 accept_multiple_files=True
        uploaded_files = st.file_uploader(
            "支持拖拽多个文件 (PDF, PPTX, DOCX, PY, TXT)", 
            type=["pdf", "pptx", "docx", "txt", "py"],
            accept_multiple_files=True 
        )
        
        if uploaded_files:
            file_count = len(uploaded_files)
            st.info(f"已选择 {file_count} 个文件等待处理")
            
            # --- 模式选择 (对这一批文件生效) ---
            st.write("###### 这一批文件的分区策略：")
            mode = st.radio("分区策略:", 
                           ["🤖 AI 智能判断", "📂 选择已有分区", "✨ 创建新分区"],
                           horizontal=True) # 横向排布更好看
            
            target_course = None
            
            if mode == "📂 选择已有分区":
                if not available_courses:
                    st.warning("暂无分区，请选择其他模式")
                    target_course = None
                else:
                    target_course = st.selectbox("选择目标:", available_courses)
            
            elif mode == "✨ 创建新分区":
                target_course = st.text_input("输入新分区名称:", placeholder="例如: 深度学习")
            
            elif mode == "🤖 AI 智能判断":
                target_course = None 
                st.caption("⚠️ 注意：大模型将分别分析每一个文件的内容来决定其归属，可能会归入不同分区。")

            # --- 批量处理按钮 ---
            if st.button(f"🚀 开始处理 ({file_count} 个文件)", use_container_width=True):
                if mode == "✨ 创建新分区" and not target_course:
                    st.error("请输入新分区名称！")
                else:
                    # 初始化进度条
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    success_count = 0
                    logs = []

                    save_dir = DATA_DIR
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)

                    # [修改 2] 循环处理文件列表
                    for i, uploaded_file in enumerate(uploaded_files):
                        # 更新状态显示
                        status_text.text(f"正在处理 ({i+1}/{file_count}): {uploaded_file.name} ...")
                        
                        try:
                            # 1. 保存文件
                            file_path = os.path.join(save_dir, uploaded_file.name)
                            with open(file_path, "wb") as f:
                                f.write(uploaded_file.getbuffer())
                            
                            # 2. 调用后端处理逻辑
                            # 注意：如果选了 AI 判断，target_course 为 None，函数内部会每张图都调一次分类器
                            result_msg = process_single_file(
                                file_path=file_path,
                                forced_course_name=target_course
                            )
                            logs.append(f"✅ {uploaded_file.name}: {result_msg}")
                            success_count += 1
                            
                        except Exception as e:
                            logs.append(f"❌ {uploaded_file.name}: 处理失败 - {str(e)}")
                        
                        # 更新进度条
                        progress_bar.progress((i + 1) / file_count)

                    # 完成后的反馈
                    status_text.text("处理完成！")
                    if success_count == file_count:
                        st.success(f"🎉 全部 {file_count} 个文件处理成功！")
                    else:
                        st.warning(f"完成 {success_count}/{file_count} 个文件，请查看下方日志。")
                    
                    # 显示详细日志
                    with st.expander("查看处理详情"):
                        for log in logs:
                            st.write(log)
                    
                    # 延时刷新
                    import time
                    time.sleep(2)
                    st.rerun()

        st.divider()
        st.caption(f"当前库中共有 {agent.vector_store.get_collection_count()} 个知识块")
# ===================== 主聊天界面 (保持不变) =====================
st.title("🎓 RAG 智能课程助教")

# 初始化消息
if "messages" not in st.session_state:
    st.session_state.messages = []

# 显示历史
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 输入框
if prompt := st.chat_input("请输入你的问题..."):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("🤔 正在思考...")
        
        # 构造历史
        history_for_agent = [
            {"role": m["role"], "content": m["content"]} 
            for m in st.session_state.messages[:-1]
        ]
        
        # 调用 Answer
        response = agent.answer_question(
            query=prompt, 
            chat_history=history_for_agent,
            course_filter=selected_course,
            top_k=selected_top_k,        # <--- 传入 Top-K
            temperature=selected_temperature # 使用 Tab 1 中选中的
        )
        
        message_placeholder.markdown(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})