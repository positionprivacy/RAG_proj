import os
from document_loader import DocumentLoader
from text_splitter import TextSplitter
from vector_store import VectorStore
from tqdm import tqdm

from config import DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP, VECTOR_DB_PATH


def main():
    if not os.path.exists(DATA_DIR):
        print(f"数据目录不存在: {DATA_DIR}")
        return

    # 1. 初始化组件
    # 注意：这里只初始化工具，不立即加载所有文档
    loader = DocumentLoader(data_dir=DATA_DIR)
    splitter = TextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    vector_store = VectorStore(db_path=VECTOR_DB_PATH)

    # [关键修改] 注释掉清空操作，实现增量更新
    vector_store.clear_collection() 
    
    print(f"当前向量库中已有 {vector_store.get_collection_count()} 条数据")

    # 2. 手动遍历目录，找出新文件
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
                
                # [关键逻辑] 检查数据库中是否已有该文件
                if vector_store.is_file_processed(file):
                    print(f"  [跳过] 已存在: {file}")
                else:
                    print(f"  [新增] 待处理: {file}")
                    files_to_process.append(file_path)

    if not files_to_process:
        print("\n所有文件均已处理，无需更新。")
        return

    # 3. 仅加载和处理新文件
    print(f"\n开始处理 {len(files_to_process)} 个新文件...")
    
    new_documents = []
    # 逐个加载新文件
    for file_path in tqdm(files_to_process, desc="加载文件"):
        # 调用 loader 加载单个文件 (这一步会触发 Qwen-VL)
        docs = loader.load_document(file_path)
        if docs:
            new_documents.extend(docs)

    if not new_documents:
        print("未提取到有效内容")
        return

    # 4. 切分文档
    chunks = splitter.split_documents(new_documents)

    # 5. 存入向量数据库 (使用 upsert)
    vector_store.add_documents(chunks)
    
    print(f"\n增量更新完成！当前库中总计 {vector_store.get_collection_count()} 条数据。")


if __name__ == "__main__":
    main()