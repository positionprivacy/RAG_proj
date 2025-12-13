import os
import io
import uuid # [新增] 用于生成唯一临时文件名
import concurrent.futures # [新增] 用于多线程并发
from typing import List, Dict, Optional
import docx2txt
from PyPDF2 import PdfReader
from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE
from openai import OpenAI

# 引入多模态需要的库
import fitz  # PyMuPDF, 用于PDF图片提取
import dashscope
from http import HTTPStatus
import config  # 导入配置文件

class DocumentLoader:
    def __init__(
        self,
        data_dir: str = config.DATA_DIR,
    ):
        self.data_dir = data_dir
        self.supported_formats = [".pdf", ".pptx", ".docx", ".txt", ".py"] 
        
        # 配置 Dashscope API (用于多模态)
        dashscope.api_key = config.OPENAI_API_KEY

        self.client = OpenAI(api_key=config.OPENAI_API_KEY, base_url=config.OPENAI_API_BASE)
        
        # [新增] 并发控制：设置最大工作线程数
        # 建议设置在 3-8 之间，防止触发 API Rate Limit (429错误)
        self.max_workers = 5

    def _describe_image(self, image_bytes: bytes, source_info: str, context_text: str = "") -> str:
        """
        [内部辅助函数] 调用 Qwen-VL 对图片进行描述
        **已修改为线程安全模式（使用唯一临时文件名）**
        """
        if not hasattr(config, "VL_MODEL_NAME") or not config.VL_MODEL_NAME:
            return ""

        # [修改] 使用 uuid 生成唯一文件名，防止多线程冲突
        unique_name = f"temp_img_{uuid.uuid4().hex}.png"

        try:
            with open(unique_name, "wb") as f:
                f.write(image_bytes)
            
            # 只有大于 5KB 的图片才处理（忽略图标、装饰线）
            if os.path.getsize(unique_name) < 5 * 1024:
                return ""

            # print(f"  > [并发] 调用 Qwen-VL: {source_info}...") 
            
            # --- 构建包含上下文的 Prompt ---
            # 截取前1000字作为背景，防止超长
            safe_context = context_text[:1000].replace("\n", " ") if context_text else "无"
            
            prompt_content = (
                f"这张图出现在课程课件中。\n"
                f"【该页面的文字内容参考】：{safe_context}\n\n"
                f"请结合上下文描述图片内容。如果图片包含具体知识点（如架构图、公式、代码截图），请详细提取文字和含义；"
                f"如果图片只是装饰或与上下文无关，请简要概括或忽略。"
            )
            # -------------------------------------

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"image": f"file://{os.path.abspath(unique_name)}"},
                        {"text": prompt_content}
                    ]
                }
            ]
            
            response = dashscope.MultiModalConversation.call(
                model=config.VL_MODEL_NAME,
                messages=messages
            )
            
            if response.status_code == HTTPStatus.OK:
                desc = response.output.choices[0].message.content[0]['text']
                # print(f"  ✔️ 图片处理成功: {source_info}")
                return f"\n[图片内容描述]: {desc}\n"
            else:
                print(f"  ! 图片处理失败 ({source_info}): {response.message}")
                return ""
                
        except Exception as e:
            print(f"  ! 图片处理异常 ({source_info}): {e}")
            return ""
        finally:
            # 清理临时文件
            if os.path.exists(unique_name):
                os.remove(unique_name)

    def _generate_summary(self, text: str) -> str:
        """
        [内部辅助函数] 使用 LLM 生成文本摘要
        保持你原有的逻辑不变
        """
        if not text.strip():
            return ""
        
        system_instruction = "你是一位专业的总结助手。请将下面的完整文件内容总结为一个简洁、准确的段落，保留核心要点和技术术语，不要增加文件内容中不存在的内容。总结内容前面加上文件涉及的课程名称/三级学科名称，总结需概括、精炼，严格保证50字以内。"
        
        # 简单截断防止超长
        text_input = text[:6000]

        prompt = system_instruction + "\n\n请基于以下内容生成摘要：\n" + text_input.strip()

        messages = [
            {"role": "user", "content": prompt} 
        ]

        try:
            response = self.client.chat.completions.create(
                model=config.MODEL_NAME, messages=messages, temperature=0.5, max_tokens=100
            )
            summary = "片段摘要：" + response.choices[0].message.content
        except Exception as e:
            print(f" ! 摘要生成失败: {e}")
            return text[:100] + ("... (摘要失败，使用前段文本)" if len(text) > 200 else "")
        
        return summary

    def load_pdf(self, file_path: str) -> List[Dict]:
        """加载PDF文件，支持多模态并发加速"""
        print(f"正在加载 PDF (并发模式): {file_path}")
        results = []
        
        reader = PdfReader(file_path)
        
        doc_fitz = None
        try:
            doc_fitz = fitz.open(file_path)
        except Exception:
            print("提示: 未安装 PyMuPDF，跳过图片识别")

        # --- 并发处理逻辑 ---
        # 1. 扫描页面，提取文本，提交图片任务
        page_tasks = [] # 存储结构: (page_text, list_of_futures)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            for i, page in enumerate(reader.pages):
                page_text = page.extract_text() or ""
                image_futures = []

                if doc_fitz and i < len(doc_fitz):
                    img_list = doc_fitz[i].get_images(full=True)
                    for img_idx, img in enumerate(img_list):
                        xref = img[0]
                        base_image = doc_fitz.extract_image(xref)
                        image_bytes = base_image["image"]
                        
                        source_info = f"Page {i+1} Img {img_idx+1}"
                        
                        # [关键] 提交任务时，传入已经提取好的 page_text 作为上下文
                        future = executor.submit(
                            self._describe_image, 
                            image_bytes, 
                            source_info, 
                            page_text 
                        )
                        image_futures.append(future)
                
                page_tasks.append((page_text, image_futures))

            if doc_fitz:
                doc_fitz.close()

            # 2. 等待结果并组装
            print(f"  > 等待 {len(page_tasks)} 页的图片解析结果...")
            
            for i, (text, futures) in enumerate(page_tasks):
                descriptions = ""
                for future in futures:
                    try:
                        descriptions += future.result() # 阻塞获取结果
                    except Exception:
                        pass
                
                full_content = f"--- 第 {i+1} 页 ---\n{text}\n{descriptions}"
                results.append({"text": full_content})
            
        return results

    def load_pptx(self, file_path: str) -> List[Dict]:
        """加载PPT文件，支持多模态并发加速"""
        print(f"正在加载 PPT (并发模式): {file_path}")
        results = []
        prs = Presentation(file_path)

        # 存储结构: (full_slide_text, list_of_futures)
        slide_tasks = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # --- Phase 1: 遍历幻灯片，提取文本，提交图片任务 ---
            for i, slide in enumerate(prs.slides):
                slide_text_parts = []
                temp_images = [] # 暂存图片对象
                
                # A. 提取所有文本 (为了做上下文)
                for shape in slide.shapes:
                    if hasattr(shape, "text_frame") and shape.text_frame:
                        slide_text_parts.append(shape.text_frame.text)
                    if shape.has_table:
                        for row in shape.table.rows:
                            row_text = " | ".join([c.text_frame.text for c in row.cells])
                            slide_text_parts.append(row_text)
                    if shape.shape_type == 13: 
                        temp_images.append(shape)
                
                full_slide_text = "\n".join(slide_text_parts)
                
                # B. 提交图片任务 (带上 full_slide_text)
                image_futures = []
                for img_shape in temp_images:
                    try:
                        image_bytes = img_shape.image.blob
                        source_info = f"Slide {i+1}"
                        future = executor.submit(
                            self._describe_image, 
                            image_bytes, 
                            source_info, 
                            full_slide_text # <--- 传入上下文
                        )
                        image_futures.append(future)
                    except:
                        pass
                
                slide_tasks.append((full_slide_text, image_futures))

            # --- Phase 2: 收集结果 ---
            print(f"  > 等待 PPT 图片解析结果...")
            
            for i, (text, futures) in enumerate(slide_tasks):
                descriptions = ""
                for future in futures:
                    try:
                        descriptions += future.result()
                    except:
                        pass

                final_text = f"--- 幻灯片 {i+1} ---\n{text}\n{descriptions}"
                results.append({"text": final_text})

        return results

    def load_docx(self, file_path: str) -> str:
        """加载DOCX文件"""
        print(f"正在加载 DOCX: {file_path}")
        text = docx2txt.process(file_path)
        return text

    def load_txt(self, file_path: str) -> str:
        """加载TXT文件 (同时也用来处理 .py 代码文件)"""
        print(f"正在加载 TXT/Code: {file_path}")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            if file_path.endswith('.py'):
                return f"```python\n{content}\n```"
            return content
        except UnicodeDecodeError:
            with open(file_path, 'r', encoding='gbk') as f:
                return f.read()

    def load_document(self, file_path: str) -> List[Dict[str, str]]:
        """
        加载单个文档
        逻辑保持不变：先加载内容(load_pdf/pptx)，再生成全篇摘要，最后分发给每页
        """
        ext = os.path.splitext(file_path)[1].lower()
        filename = os.path.basename(file_path)
        documents = []
        content_chunks = []
        if ext == ".pdf":
            # load_pdf 现在内部是并发的，返回 list[dict]
            pages = self.load_pdf(file_path)
            
            # 拼合全文以生成摘要
            over_all_text = ""
            for p in pages:
                over_all_text += p["text"] + "\n"
            summary = self._generate_summary(over_all_text)

            for i, p in enumerate(pages, 1):
                documents.append({
                    "content": p["text"],
                    "filename": filename,
                    "filepath": file_path,
                    "filetype": ext,
                    "page_number": i,
                    "summary": summary
                })
            return documents
            
        elif ext == ".pptx":
            # load_pptx 现在内部是并发的
            slides = self.load_pptx(file_path)
            
            over_all_text = ""
            for s in slides:
                over_all_text += s["text"] + "\n"
            summary = self._generate_summary(over_all_text)

            for i, s in enumerate(slides, 1):
                documents.append({
                    "content": s["text"],
                    "filename": filename,
                    "filepath": file_path,
                    "filetype": ext,
                    "page_number": i,
                    "summary": summary
                })
            return documents
        
        elif ext == ".docx":
            content = self.load_docx(file_path)
        elif ext == ".txt" or ext == ".py":
            content = self.load_txt(file_path)
        else:
            print(f"不支持的文件格式: {ext}")
            return []

        if content:
            summary = self._generate_summary(content)
            documents.append({
                "content": content,
                "filename": filename,
                "filepath": file_path,
                "filetype": ext,
                "page_number": 1,
                "summary": summary 
            })
            
        return documents

    def load_all_documents(self) -> List[Dict[str, str]]:
        """加载数据目录下的所有文档"""
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir, exist_ok=True)
            print(f"数据目录不存在，已创建空目录: {self.data_dir}")
            return []

        documents = []
        for root, dirs, files in os.walk(self.data_dir):
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in self.supported_formats:
                    file_path = os.path.join(root, file)
                    doc_chunks = self.load_document(file_path)
                    if doc_chunks:
                        documents.extend(doc_chunks)

        print(f"所有文档加载完毕，共处理 {len(documents)} 个片段。")
        return documents