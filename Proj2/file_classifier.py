import os
from typing import List
from openai import OpenAI
from config import OPENAI_API_KEY, OPENAI_API_BASE, MODEL_NAME

class FileClassifier:
    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_API_BASE)
        self.model = MODEL_NAME

    def determine_course(self, filename: str, content_preview: str, existing_courses: List[str]) -> str:
        """
        根据文件名、内容预览以及【现有课程列表】，判断课程名称。
        实现实体对齐：优先归入已有课程。
        """
        # 截取前 1500 个字符作为预览，节省 Token
        preview = content_preview[:1500]
        
        # 构造提示词，注入"记忆"
        if existing_courses:
            courses_str = ", ".join([f'"{c}"' for c in existing_courses])
            context_instruction = f"""
            目前数据库中已有的课程分区为：[{courses_str}]。
            
            **关键规则（实体对齐）**：
            1. 请首先判断该文件是否属于上述【已有课程】中的某一个。
            2. 如果是（例如文件是"CV_Homework.pdf"，而已有课程中有"计算机视觉"），请**必须直接返回已有课程的名称**（即返回"计算机视觉"）。
            3. 只有当该文件确实属于一门全新的、列表中不存在的课程时，才输出一个新的规范名称。
            """
        else:
            context_instruction = "目前数据库为空，你可以根据文件内容创建一个规范的课程名称（例如：'计算机视觉'、'操作系统'）。"

        prompt = f"""
        你是一个课程资料管理员。请分析以下文件信息，判断它属于哪门课程。
        
        {context_instruction}
        
        待分析文件名: {filename}
        文件内容预览:
        {preview}
        ...
        
        要求：
        1. 输出一个简洁的课程名称。
        2. 不要输出任何解释性文字，只输出课程名称字符串。
        3. 去除名称中的特殊符号。
        """

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1, # 极低温度，保证分类稳定
            )
            course_name = response.choices[0].message.content.strip()
            # 简单的清洗，防止模型输出标点
            return course_name.replace("。", "").replace("！", "").replace("'", "").replace('"', "").replace(":", "")
        except Exception as e:
            print(f"  ! 分类失败: {e}")
            return "通用课程"