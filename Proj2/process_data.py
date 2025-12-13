import os
from document_loader import DocumentLoader
from text_splitter import TextSplitter
from vector_store import VectorStore
from file_classifier import FileClassifier # [新增]
from tqdm import tqdm
from typing import Optional
from config import DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP, VECTOR_DB_PATH
def process_single_file(
    file_path: str, 
    forced_course_name: Optional[str] = None
) -> str:
    """
    处理单个文件的核心逻辑（供 Web 端调用）
    
    参数:
        file_path: 文件路径
        forced_course_name: 
            - None: 让 AI 自动判断
            - "xxx": 强制归类为 xxx 课程
            
    返回:
        处理结果消息
    """
    # 1. 初始化组件
    loader = DocumentLoader(data_dir=os.path.dirname(file_path))
    splitter = TextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    vector_store = VectorStore(db_path=VECTOR_DB_PATH)
    classifier = FileClassifier()

    filename = os.path.basename(file_path)
    
    # 检查是否已存在
    if vector_store.is_file_processed(filename):
        return f"文件 {filename} 已存在于知识库中，已跳过。"

    # 2. 加载文档
    docs = loader.load_document(file_path)
    if not docs:
        return f"文件 {filename} 加载失败或内容为空。"

    # 3. 确定课程名称
    final_course_name = "未知课程"
    
    if forced_course_name:
        # A. 用户手动指定/创建
        final_course_name = forced_course_name
        print(f"  [Manual] 强制归类为: {final_course_name}")
    else:
        # B. AI 自动判断
        preview_text = docs[0]['content']
        # 获取现有课程作为记忆
        current_courses = vector_store.get_all_courses()
        final_course_name = classifier.determine_course(
            filename, preview_text, list(current_courses)
        )
        print(f"  [AI Auto] 自动归类为: {final_course_name}")

    # 4. 写入元数据
    for doc in docs:
        doc['course_name'] = final_course_name

    # 5. 切分
    chunks = splitter.split_documents(docs)

    # 6. 存入
    vector_store.add_documents(chunks)
    
    return f"成功处理 {filename}，归入分区：[{final_course_name}]，生成 {len(chunks)} 个知识块。"
def main():
    if not os.path.exists(DATA_DIR):
        print(f"数据目录不存在: {DATA_DIR}")
        return

    # 1. 初始化组件
    loader = DocumentLoader(data_dir=DATA_DIR)
    splitter = TextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    vector_store = VectorStore(db_path=VECTOR_DB_PATH)
    classifier = FileClassifier() # [新增]

    # [关键] 获取当前已有的课程列表，作为"记忆"
    # 使用 set 方便快速去重
    current_courses = set(vector_store.get_all_courses())
    print(f"🔍 当前向量库中已存在的课程分区: {current_courses if current_courses else '无 (冷启动)'}")

    # 2. 扫描文件
    supported_formats = [".pdf", ".pptx", ".docx", ".txt", ".py"]
    files_to_process = []

    print("正在扫描新文件...")
    for root, dirs, files in os.walk(DATA_DIR):
        for file in files:
            if file.startswith("~$") or file.startswith("."):
                continue
            ext = os.path.splitext(file)[1].lower()
            if ext in supported_formats:
                file_path = os.path.join(root, file)
                
                if vector_store.is_file_processed(file):
                    print(f"  [跳过] 已存在: {file}")
                else:
                    print(f"  [新增] 待处理: {file}")
                    files_to_process.append(file_path)

    if not files_to_process:
        print("\n所有文件均已处理，无需更新。")
        return

    # 3. 逐个处理新文件 (加载 -> 分类 -> 标记)
    print(f"\n开始处理 {len(files_to_process)} 个新文件...")
    
    new_documents = []
    
    # 我们逐个处理，以便实时更新 current_courses
    for file_path in tqdm(files_to_process, desc="加载并分类"):
        # A. 加载文档
        docs = loader.load_document(file_path)
        if not docs: continue
        
        # B. 智能分类
        filename = os.path.basename(file_path)
        preview_text = docs[0]['content'] # 取第一页/第一块内容预览
        
        # 调用分类器，传入当前的记忆
        course_name = classifier.determine_course(
            filename=filename, 
            content_preview=preview_text, 
            existing_courses=list(current_courses)
        )
        
        # C. 更新记忆 (这样这一批次后续的文件就能看到这个新名字)
        if course_name not in current_courses:
            print(f"\n  ✨ 新建分区: [{course_name}] (文件: {filename})")
            current_courses.add(course_name)
        else:
            # 可以在 tqdm 进度条外打印，避免刷屏
            pass 

        # D. 将课程名打入 Metadata
        for doc in docs:
            doc['course_name'] = course_name
            
        new_documents.extend(docs)

    if not new_documents:
        print("未提取到有效内容")
        return

    # 4. 切分文档 (此时 docs 里已经有了 course_name)
    chunks = splitter.split_documents(new_documents)

    # 5. 存入向量数据库
    vector_store.add_documents(chunks)
    
    print(f"\n✅ 处理完成！当前库中总计 {vector_store.get_collection_count()} 条数据。")
    print(f"📚 当前所有课程分区: {list(current_courses)}")

if __name__ == "__main__":
    main()