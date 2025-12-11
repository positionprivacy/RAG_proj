# test_document_loader.py

import os
import tempfile
from pathlib import Path
from document_loader import DocumentLoader

DATA_DIR = "data"

def test_document_loader():
    """测试DocumentLoader类"""
    print("=" * 50)
    print("开始测试DocumentLoader")
    print("=" * 50)

    pdf_path = os.path.join(DATA_DIR, "sample.pdf")
    txt_path = os.path.join(DATA_DIR, "sample.txt")
    docx_path = os.path.join(DATA_DIR, "sample.docx")
    pptx_path = os.path.join(DATA_DIR, "sample.pptx")

    for file in os.listdir(DATA_DIR):
        if file.endswith((".pdf")):
            pdf_path = os.path.join(DATA_DIR, file)
        elif file.endswith((".txt")):
            txt_path = os.path.join(DATA_DIR, file)
        elif file.endswith((".docx")):
            docx_path = os.path.join(DATA_DIR, file)
        elif file.endswith((".pptx")):
            pptx_path = os.path.join(DATA_DIR, file)
            
    # 创建DocumentLoader实例
    loader = DocumentLoader(data_dir=DATA_DIR)
    
    # 测试单个文件加载
    print("\n1. 测试PDF文件加载:")
    if os.path.exists(pdf_path):
        pdf_docs = loader.load_document(pdf_path)
        print(f"  加载到 {len(pdf_docs)} 个页面")
        for i, doc in enumerate(pdf_docs[:2]):  # 只显示前两个页面
            print(f"  页面 {doc['page_number']}: {doc['content'][:50]}...")
    
    print("\n2. 测试TXT文件加载:")
    if os.path.exists(txt_path):
        txt_docs = loader.load_document(txt_path)
        print(f"  加载到 {len(txt_docs)} 个文档块")
        if txt_docs:
            print(f"  内容预览: {txt_docs[0]['content'][:100]}...")
    
    print("\n3. 测试DOCX文件加载:")
    if docx_path and os.path.exists(docx_path):
        docx_docs = loader.load_document(docx_path)
        print(f"  加载到 {len(docx_docs)} 个文档块")
        if docx_docs:
            print(f"  内容预览: {docx_docs[0]['content'][:100]}...")
    
    print("\n4. 测试PPTX文件加载:")
    if pptx_path and os.path.exists(pptx_path):
        pptx_docs = loader.load_document(pptx_path)
        print(f"  加载到 {len(pptx_docs)} 个幻灯片")
        for i, doc in enumerate(pptx_docs[:2]):  # 只显示前两个幻灯片
            print(f"  幻灯片 {doc['page_number']}: {doc['content'][:50]}...")
    
    # 测试批量加载
    print("\n5. 测试批量加载所有文档:")
    all_docs = loader.load_all_documents()
    print(f"  总共加载了 {len(all_docs) if all_docs else 0} 个文档块")
    
    # 显示加载的文档信息
    if all_docs:
        print("\n6. 加载的文档块信息:")
        for i, doc in enumerate(all_docs[:3]):  # 只显示前三个文档块
            print(f"  文档块 {i+1}:")
            print(f"    文件名: {doc['filename']}")
            print(f"    文件类型: {doc['filetype']}")
            print(f"    页码: {doc['page_number']}")
            print(f"    内容长度: {len(doc['content'])} 字符")
            print(f"    内容预览: {doc['content'][:80].replace(chr(10), ' ')}...")
    
   
    print("\n" + "=" * 50)
    print("测试完成")
    print("=" * 50)



if __name__ == "__main__":
    # 运行所有测试
    test_document_loader()
