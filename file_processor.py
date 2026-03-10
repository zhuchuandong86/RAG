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

def parse_file_to_md(file_path):
    if is_duplicate(file_path): return []
    ext = os.path.splitext(file_path)[-1].lower()
    docs = []
    
# --- 新增：中间 MD 保存逻辑 ---
    debug_dir = "01_RAG\data\debug_md"
    os.makedirs(debug_dir, exist_ok=True)
    # 获取文件名（不含路径和后缀）
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    debug_file_path = os.path.join(debug_dir, f"{base_name}.md")

    if ext in ['.docx', '.doc']:
        d = DocxDocument(file_path)
        t = text_to_md("\n".join([p.text for p in d.paragraphs]))
        docs.append(Document(page_content=t, metadata={"source": file_path}))
    elif ext in ['.jpg', '.png', '.jpeg']:
        t = image_to_md_via_vlm(file_path)
        docs.append(Document(page_content=t, metadata={"source": file_path}))
    elif ext == '.pdf':
        pdf = fitz.open(file_path)
        for i in range(len(pdf)):
            page = pdf[i]
            raw = page.get_text("text").strip()
            if len(raw) < 50: # 识别扫描页
                pix = page.get_pixmap(dpi=150)
                tmp = f"temp_{i}.jpg"
                pix.save(tmp)
                t = image_to_md_via_vlm(tmp)
                os.remove(tmp)
            else:
                t = text_to_md(raw)
            docs.append(Document(page_content=t, metadata={"source": file_path, "page": i+1}))
        pdf.close()
    return docs
