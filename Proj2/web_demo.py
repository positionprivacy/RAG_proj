import streamlit as st
import os
import uuid
import shutil
from datetime import datetime
import base64
import re
from rag_agent import RAGAgent
from config import MODEL_NAME, DATA_DIR
from document_loader import DocumentLoader 
from process_data import process_single_file 

# ================= 1. 初始化与配置 =================
st.set_page_config(page_title="智能课程助教", page_icon="🎓", layout="wide")

@st.cache_resource
def init_resources():
    agent = RAGAgent(model=MODEL_NAME)
    loader = DocumentLoader(data_dir=DATA_DIR)
    return agent, loader

agent, loader = init_resources()

# --- 状态管理初始化 ---
if "sessions" not in st.session_state:
    st.session_state.sessions = {}
    new_id = str(uuid.uuid4())
    st.session_state.sessions[new_id] = {
        "title": "新对话",
        "messages": [],
        "created_at": datetime.now()
    }
    st.session_state.current_session_id = new_id

if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

def get_current_session():
    return st.session_state.sessions[st.session_state.current_session_id]

def create_new_session():
    new_id = str(uuid.uuid4())
    st.session_state.sessions[new_id] = {
        "title": f"对话 {datetime.now().strftime('%H:%M')}",
        "messages": [],
        "created_at": datetime.now()
    }
    st.session_state.current_session_id = new_id
    st.session_state.uploader_key += 1

def delete_session(sid):
    if len(st.session_state.sessions) > 1:
        del st.session_state.sessions[sid]
        if st.session_state.current_session_id == sid:
            st.session_state.current_session_id = list(st.session_state.sessions.keys())[0]
        st.rerun()

# ================= 2. 核心工具函数 (原生修复版) =================

# [修改] 使用专用库渲染 PDF
def display_pdf_at_page(file_path: str, page_number: int):
    """
    使用 streamlit-pdf-viewer 渲染 PDF。
    它能完美解决浏览器拦截 Base64 的问题。
    """
    try:
        from streamlit_pdf_viewer import pdf_viewer
        
        # 1. 检查文件
        if not os.path.exists(file_path):
            st.error("文件不存在")
            return

        # 2. 渲染
        # input: 直接传文件路径
        # width: 宽度
        # height: 高度 (虽然这个库主要靠宽度控制)
        # pages_to_render: [page_number] (只渲染特定页，如果要看全本，去掉这个参数)
        # 如果你想让用户能滚动看全本，但自动跳到第N页，目前这个库支持 scroll_to_page (新版)
        # 这里我们先只渲染引用页，保证速度
        
        st.caption(f"正在渲染第 {page_number} 页...")
        
        pdf_viewer(
            input=file_path, 
            width=700, 
            height=800,
            pages_to_render=[page_number] # 只显示这一页，速度极快
        )
        
    except ImportError:
        st.error("请先安装依赖库: pip install streamlit-pdf-viewer")
    except Exception as e:
        st.error(f"渲染失败: {e}")

def render_references(response_text: str, key_suffix: str):
    """
    [终极容错版] 扫描文件名 -> 显示下载 -> 调用 PDF 渲染
    """
    if not isinstance(response_text, str) or not os.path.exists(DATA_DIR):
        return

    all_files = [f for f in os.listdir(DATA_DIR) if os.path.isfile(os.path.join(DATA_DIR, f))]
    found_refs = [] 
    
    # 扫描文件名
    for filename in all_files:
        if filename in response_text:
            # 找页码
            page_pattern = re.escape(filename) + r"[^\d]{0,20}(\d+)"
            page_match = re.search(page_pattern, response_text)
            page_num = 1
            if page_match:
                page_num = int(page_match.group(1))
            found_refs.append((filename, page_num))
    
    if not found_refs: return

    st.markdown("---")
    st.caption(f"📚 **检测到 {len(found_refs)} 个参考文件**")
    
    tabs = st.tabs([f"📄 {name[:10]}.." for name, _ in found_refs])
    
    for i, (filename, page_num) in enumerate(found_refs):
        with tabs[i]:
            file_path = os.path.join(DATA_DIR, filename)
            unique_key = f"{key_suffix}_{i}_{filename}"
            ext = os.path.splitext(filename)[1].lower()

            col1, col2 = st.columns([0.3, 0.7])
            
            with col1:
                st.markdown(f"**文件**: `{filename}`")
                st.markdown(f"**页码**: `{page_num}`")
                try:
                    with open(file_path, "rb") as f:
                        st.download_button("⬇️ 下载文件", f, file_name=filename, key=f"dl_{unique_key}")
                except: st.error("文件读取失败")

            with col2:
                if ext == ".pdf":
                    # 调用新写的函数
                    display_pdf_at_page(file_path, page_num)
                elif ext in [".png", ".jpg", ".jpeg"]:
                    st.image(file_path)
                elif ext in [".txt", ".py", ".cpp", ".c", ".java"]:
                    with open(file_path, "r", encoding='utf-8') as f:
                        st.code(f.read(2000), language=ext[1:])
                else:
                    st.info("请下载查看。")

# ================= 3. 侧边栏 =================
with st.sidebar:
    st.title("🎓 助教控制台")
    
    tab_chat, tab_kb = st.tabs(["💬 对话设置", "📂 知识库管理"])
    
    with tab_chat:
        st.caption("会话列表")
        if st.button("➕ 新建对话", use_container_width=True):
            create_new_session()
            st.rerun()
            
        session_ids = list(st.session_state.sessions.keys())
        for sid in reversed(session_ids):
            sess = st.session_state.sessions[sid]
            col1, col2 = st.columns([0.85, 0.15])
            label = f"{'🟢 ' if sid == st.session_state.current_session_id else ''}{sess['title']}"
            if col1.button(label, key=sid, use_container_width=True):
                st.session_state.current_session_id = sid
                st.rerun()
            if col2.button("✖", key=f"del_{sid}"):
                delete_session(sid)

        st.divider()
        st.subheader("⚙️ 参数设置")
        available_courses = agent.vector_store.get_all_courses()
        options = ["全局搜索"] + available_courses
        selected_course = st.selectbox("选择检索分区:", options, index=0)
        selected_top_k = st.slider("检索数量 (Top-K):", 1, 10, 3)
        selected_temperature = st.slider("创造力 (Temp):", 0.0, 1.5, 0.7)

    with tab_kb:
        st.subheader("📤 批量入库")
        uploaded_files = st.file_uploader("拖拽文件入库", type=["pdf", "pptx", "docx", "txt", "py"], accept_multiple_files=True)
        if uploaded_files:
            st.write(f"已选 {len(uploaded_files)} 个文件")
            mode = st.radio("分区策略:",["🤖 AI 智能判断", "📂 选择已有分区", "✨ 创建新分区"], horizontal=True, label_visibility="collapsed")
            target_course = None
            if mode == "📂 选择已有分区":
                target_course = st.selectbox("目标:", available_courses) if available_courses else None
            elif mode == "✨ 创建新分区":
                target_course = st.text_input("新分区名:")
            
            if st.button(f"🚀 开始存入", use_container_width=True):
                progress_bar = st.progress(0)
                if not os.path.exists(DATA_DIR): os.makedirs(DATA_DIR)
                for i, f in enumerate(uploaded_files):
                    try:
                        file_path = os.path.join(DATA_DIR, f.name)
                        with open(file_path, "wb") as w: w.write(f.getbuffer())
                        process_single_file(file_path, forced_course_name=target_course)
                    except: pass
                    progress_bar.progress((i + 1) / len(uploaded_files))
                st.success("入库完成！")
                import time; time.sleep(1); st.rerun()
        st.divider()
        st.caption(f"当前库中片段数: {agent.vector_store.get_collection_count()}")

# ================= 4. 主界面：聊天窗口 =================
current_sess = get_current_session()
st.header(current_sess["title"])

# --- A. 显示历史消息 ---
for i, msg in enumerate(current_sess["messages"]):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("attachment"):
            st.caption(f"📎 附件: {msg['attachment']}")
        
        # [渲染引用] 使用唯一 Key
        if msg["role"] == "assistant":
            render_references(msg["content"], key_suffix=f"history_{i}")

# --- B. 附件上传 ---
with st.container():
    with st.popover("📎 添加附件 (图片/文件)", use_container_width=True):
        st.caption("临时文件，仅本轮有效。")
        chat_file = st.file_uploader(
            "支持 PNG, JPG, PDF, PPTX, DOCX, PY", 
            type=["png", "jpg", "jpeg", "pdf", "pptx", "docx", "txt", "py"],
            key=f"chat_uploader_{st.session_state.uploader_key}" 
        )

# --- C. 输入处理 ---
if prompt := st.chat_input("请输入你的问题..."):
    
    file_context = ""
    attachment_name = None
    
    if chat_file:
        attachment_name = chat_file.name
        if not os.path.exists("temp_uploads"): os.makedirs("temp_uploads")
        temp_path = os.path.join("temp_uploads", chat_file.name)
        with open(temp_path, "wb") as f: f.write(chat_file.getbuffer())
        
        with st.status("正在解析附件...", expanded=False):
            parsed_text = loader.parse_file_to_text(temp_path)
        file_context = f"\n\n--- 临时文件 [{attachment_name}] ---\n{parsed_text}\n----------------\n"

    final_query = prompt
    if file_context:
        final_query = f"{file_context}\n\n用户问题: {prompt}"

    current_sess["messages"].append({"role": "user", "content": prompt, "attachment": attachment_name})
    
    with st.chat_message("user"):
        st.markdown(prompt)
        if attachment_name: st.caption(f"📎 已上传: {attachment_name}")

    with st.chat_message("assistant"):
        placeholder = st.empty()
        
        history_pure = [{"role": m["role"], "content": m["content"]} for m in current_sess["messages"][:-1]]
        
        with st.status("🤖 助教正在思考...", expanded=True) as status:
            response_data = agent.answer_question(
                query=final_query,
                chat_history=history_pure,
                course_filter=selected_course,
                top_k=selected_top_k,
                temperature=selected_temperature
            )
            if isinstance(response_data, dict) and "traces" in response_data:
                for t in response_data["traces"]: st.write(t)
            status.update(label="✅ 回答生成完毕", state="complete", expanded=False)
        
        if isinstance(response_data, dict):
            final_response = response_data.get("response", "生成失败")
        else:
            final_response = response_data
            
        placeholder.markdown(final_response)
        
        # [渲染引用]
        render_references(final_response, key_suffix=f"current_{len(current_sess['messages'])}")
    
    current_sess["messages"].append({"role": "assistant", "content": final_response})
    
    if len(current_sess["messages"]) == 2:
        with st.status("正在生成标题...", expanded=False):
            current_sess["title"] = agent.generate_session_title(prompt)
        st.rerun()
    
    st.session_state.uploader_key += 1
    st.rerun()