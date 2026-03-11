import os
import json
import hashlib
import fitz # PyMuPDF
from docx import Document as DocxDocument
from langchain_core.documents import Document
from md_converter import text_to_md, image_to_md_via_vlm
from config import Config

def get_file_md5(file_path):
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def is_duplicate(file_path):
    file_md5 = get_file_md5(file_path)
    records = {}
    
    # 核心修复：确保父目录存在
    dir_name = os.path.dirname(Config.PROCESSED_RECORD_FILE)
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)
        print(f"已自动创建缺失的目录: {dir_name}") # 建议商用取消

    if os.path.exists(Config.PROCESSED_RECORD_FILE):
        with open(Config.PROCESSED_RECORD_FILE, "r", encoding="utf-8") as f:
            records = json.load(f)
    
    if file_md5 in records:
        return True
    
    records[file_md5] = file_path
    with open(Config.PROCESSED_RECORD_FILE, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    return False

# file_processor.py 核心修改片段
def parse_file_to_md(file_path):
    # 1. 查重逻辑（稍后在 batch_ingest 修改，这里返回内容）
    ext = os.path.splitext(file_path)[-1].lower()
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    docs = []
    full_md_content = f"# 文件标题：{base_name}\n\n" # 引用文件名作为大标题

    if ext == '.pdf':
        pdf = fitz.open(file_path)
        for i in range(len(pdf)):
            page = pdf[i]
            # ✨ 增加表格检测：如果页面包含表格或图像，强制走 VLM 以确保表格对齐
            has_tables = len(page.find_tables().tables) > 0
            has_images = len(page.get_images()) > 0
            
            if has_tables or has_images or len(page.get_text().strip()) < 50:
                print(f"第 {i+1} 页发现表格或图像，调用 VLM 进行结构化解析...")
                pix = page.get_pixmap(dpi=150)
                tmp = f"temp_{i}.jpg"
                pix.save(tmp)
                t = image_to_md_via_vlm(tmp) # VLM 会自动处理表格对齐
                os.remove(tmp)
            else:
                t = text_to_md(page.get_text())
            
            full_md_content += f"## 第 {i+1} 页\n{t}\n\n"
            docs.append(Document(page_content=t, metadata={"source": file_path, "title": base_name, "page": i+1}))
        pdf.close()

    # 2. 无论什么格式，同步存一个 MD 文件
    debug_dir = os.path.join("01_RAG", "data", "debug_md")
    os.makedirs(debug_dir, exist_ok=True)
    with open(os.path.join(debug_dir, f"{base_name}.md"), "w", encoding="utf-8") as f:
        f.write(full_md_content)
    
    return docs
