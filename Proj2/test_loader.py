from document_loader import DocumentLoader
import time

def test():
    print("========== 开始测试 DocumentLoader ==========")
    
    # 初始化加载器
    loader = DocumentLoader()
    
    # 记录开始时间
    start_time = time.time()
    
    # 执行加载
    print("正在扫描 data/ 目录并加载文档...")
    documents = loader.load_all_documents()
    
    end_time = time.time()
    
    print(f"\n========== 加载完成 ==========")
    print(f"耗时: {end_time - start_time:.2f} 秒")
    
    if not documents:
        print("❌ 警告: 没有加载到任何文档！请检查 data/ 目录下是否有文件。")
        return

    print(f"✅ 成功加载文档片段数: {len(documents)}")
    
    # 打印前几个文档的详细信息，检查内容
    print("\n========== 内容抽查 ==========")
    for i, doc in enumerate(documents[:3]): # 只看前3个片段
        print(f"\n[片段 {i+1}]")
        print(f"📄 文件名: {doc['filename']}")
        print(f"📑 页码/位置: {doc['page_number']}")
        print(f"📏 字符数: {len(doc['content'])}")
        
        # 重点检查：是否包含了多模态描述？
        if "[图片内容描述]" in doc['content']:
            print("✨ 发现图片描述 (Qwen-VL 工作正常!)")
        
        # 重点检查：Python代码是否被包裹？
        if doc['filename'].endswith('.py'):
            if "```python" in doc['content']:
                print("🐍 Python代码格式化正常")
            else:
                print("❌ Python代码未检测到 Markdown 标记")

        # 打印内容预览 (前200字符)
        preview = doc['content'][:200].replace('\n', ' ')
        print(f"📝 内容预览: {preview}...")
        print("-" * 50)

if __name__ == "__main__":
    test()