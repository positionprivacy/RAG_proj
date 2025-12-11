from typing import List, Dict
import re
from tqdm import tqdm

class TextSplitter:
    def __init__(self, chunk_size: int, chunk_overlap: int):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # 定义分句的优先级：双换行 > 单换行 > 中文句号/叹号/问号 > 英文句号/叹号/问号
        # 注意：代码文件 (.py) 主要依赖换行符，且要注意不要切断 obj.method() 中的点
        self.separators = ["\n\n", "\n", "。", "！", "？", "!", "?", ";"]

    def split_text(self, text: str) -> List[str]:
        """将文本切分为块
        
        策略：
        1. 尝试截取 chunk_size 长度的文本。
        2. 如果该位置不是句子结尾，向后回溯查找最近的分隔符。
        3. 保证切分点位于完整句子的边界。
        4. 应用 chunk_overlap 保持上下文。
        """
        if not text:
            return []

        chunks = []
        start = 0
        text_len = len(text)

        while start < text_len:
            # 1. 确定粗略的结束位置 (硬切分点)
            end = start + self.chunk_size
            
            # 如果剩余文本不足一个 chunk，直接取完并结束
            if end >= text_len:
                chunks.append(text[start:])
                break

            # 2. 优化结束位置：寻找最近的句子边界 (软切分点)
            # 我们从硬切分点 end 开始向前回溯，寻找分隔符
            best_split = end
            found_separator = False
            
            # 限制回溯范围，防止回溯太多导致 chunk 太短 (例如只回溯 20%)
            search_limit = max(start, end - int(self.chunk_size * 0.2))

            # 从 end 倒序遍历到 search_limit
            for i in range(end, search_limit, -1):
                # 检查当前字符是否是分隔符
                # 注意：我们切分在分隔符之后 (i+1)，保留标点符号在上一句
                if i < text_len and text[i] in self.separators:
                    # 特殊判断：如果是英文点号 '.'，要防止切分 3.14 或 obj.func()
                    if text[i] in ['.']:
                        # 如果前后都是数字或字母，大概率不是句号，跳过
                        if (i > 0 and i < text_len - 1 and 
                            text[i-1].isalnum() and text[i+1].isalnum()):
                            continue
                            
                    best_split = i + 1
                    found_separator = True
                    break
            
            # 3. 如果没找到合适的分隔符，就强制在 end 处切分
            if not found_separator:
                best_split = end
            
            # 4. 加入结果
            chunk = text[start:best_split]
            chunks.append(chunk)

            # 5. 计算下一个 chunk 的起始位置 (移动步长 = 长度 - 重叠)
            # 下一次的 start 应该是当前结束位置 - overlap
            # 这样新的 chunk 就会包含上一块结尾的 overlap 部分
            start = best_split - self.chunk_overlap
            
            # 防止死循环：如果 overlap 大于 chunk 长度，强制前进一步
            if start >= best_split:
                start = best_split

        return chunks

    def split_documents(self, documents: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """切分多个文档"""
        chunks_with_metadata = []

        for doc in tqdm(documents, desc="处理文档", unit="文档"):
            content = doc.get("content", "")
            filetype = doc.get("filetype", "")
            filename = doc.get("filename", "unknown")

            # 策略 A: 对于 PDF 和 PPT，老师要求不再二次切分 (按页/幻灯片)
            # 优点：保持页面完整性，利于引用 "第几页"
            # 风险：如果某一页字数特别多（加上Qwen-VL的描述后），可能超过模型窗口。
            # 这里我们遵循老师模板，不做切分。
            if filetype in [".pdf", ".pptx"]:
                chunk_data = {
                    "content": doc.get("summary", "") + content,
                    "filename": filename,
                    "filepath": doc.get("filepath", ""),
                    "filetype": filetype,
                    "page_number": doc.get("page_number", 0),
                    "chunk_id": 0, # 这里不切分，ID默认为0
                }
                chunks_with_metadata.append(chunk_data)

            # 策略 B: 对于纯文本、Word 文档和代码文件，进行滑动窗口切分
            # 注意：我在这里加上了 .py，因为代码文件需要被切分
            elif filetype in [".docx", ".txt", ".py"]:
                chunks = self.split_text(content)
                for i, chunk in enumerate(chunks):
                    chunk_data = {
                        "content": doc.get("summary", "") + chunk,
                        "filename": filename,
                        "filepath": doc.get("filepath", ""),
                        "filetype": filetype,
                        "page_number": 0, # TXT/Word通常没有页码概念
                        "chunk_id": i,    # 记录切分块的序号，检索时可用于排序
                    }
                    chunks_with_metadata.append(chunk_data)

        print(f"\n文档处理完成，共生成 {len(chunks_with_metadata)} 个知识块")
        return chunks_with_metadata