import os
import json
import hashlib
import fitz 
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

# 💡 修正 1：只检查，不记录
def check_duplicate(file_path):
    """只查重，不写入"""
    file_md5 = get_file_md5(file_path)
    if not os.path.exists(Config.PROCESSED_RECORD_FILE):
        return False
    with open(Config.PROCESSED_RECORD_FILE, "r", encoding="utf-8") as f:
        records = json.load(f)
    return file_md5 in records

def mark_as_processed(file_path):
    """解析成功后，才将其登记入册"""
    file_md5 = get_file_md5(file_path)
    records = {}
    if os.path.exists(Config.PROCESSED_RECORD_FILE):
        with open(Config.PROCESSED_RECORD_FILE, "r", encoding="utf-8") as f:
            records = json.load(f)
            
    records[file_md5] = file_path
    with open(Config.PROCESSED_RECORD_FILE, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

def parse_file_to_md(file_path):
    ext = os.path.splitext(file_path)[-1].lower()
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    docs = []
    full_md_content = f"# 文档标题：{base_name}\n\n"

    try:
        if ext == '.pdf':
            pdf = fitz.open(file_path)
            for i in range(len(pdf)):
                page = pdf[i]
                has_tables = len(page.find_tables().tables) > 0
                has_images = len(page.get_images()) > 0
                
                if has_tables or has_images or len(page.get_text().strip()) < 50:
                    pix = page.get_pixmap(dpi=150)
                    tmp = f"temp_{i}.jpg"
                    pix.save(tmp)
                    t = image_to_md_via_vlm(tmp)
                    if os.path.exists(tmp): os.remove(tmp)
                else:
                    t = text_to_md(page.get_text())
                
                full_md_content += f"## 第 {i+1} 页\n{t}\n\n"
                docs.append(Document(page_content=t, metadata={"source": file_path, "title": base_name, "page": i+1}))
            pdf.close()
            
        elif ext in ['.docx', '.doc']:
            d = DocxDocument(file_path)
            t = text_to_md("\n".join([p.text for p in d.paragraphs]))
            full_md_content += t
            docs.append(Document(page_content=t, metadata={"source": file_path}))
            
    except Exception as e:
        print(f"解析异常: {e}")
        return [] # 解析失败，直接返回空列表

    debug_dir = os.path.join( "data", "debug_md")
    os.makedirs(debug_dir, exist_ok=True)
    with open(os.path.join(debug_dir, f"{base_name}.md"), "w", encoding="utf-8") as f:
        f.write(full_md_content)
        
    return docs
