from langchain_text_splitters import MarkdownTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from file_processor import check_duplicate, mark_as_processed, parse_file_to_md
from config import Config
import os
import shutil
import json
import pickle

def batch_ingest_folder(folder_path: str):
    all_docs = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            if check_duplicate(file_path): continue
            
            file_docs = parse_file_to_md(file_path)
            if file_docs: # 只有真正解析出了内容
                all_docs.extend(file_docs)
                mark_as_processed(file_path) # 此时才写入 JSON 登记册

    if not all_docs: return

    text_splitter = MarkdownTextSplitter(chunk_size=Config.CHUNK_SIZE, chunk_overlap=Config.CHUNK_OVERLAP)
    chunks = text_splitter.split_documents(all_docs)
    
    if not os.path.exists(Config.DB_DIR): os.makedirs(Config.DB_DIR)

    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = Config.RETRIEVER_TOP_K
    with open(os.path.join(Config.DB_DIR, "bm25_index.pkl"), "wb") as f:
        pickle.dump(bm25_retriever, f)

    embeddings = OpenAIEmbeddings(model=Config.EMBEDDING_MODEL, api_key=Config.INTERNAL_API_KEY, base_url=Config.INTERNAL_BASE_URL, check_embedding_ctx_length=False)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(Config.DB_DIR)

# batch_ingest.py
# 顶部导入需要更新：
from file_processor import get_file_md5, check_duplicate, mark_as_processed, parse_file_to_md

def ingest_single_file(file_path, force_overwrite=False):
    """支持覆盖逻辑的单文件入库"""
    # 修复：原代码使用未定义的 is_duplicate，应改为 check_duplicate
    if not force_overwrite and check_duplicate(file_path):
        return "EXISTS"

    # 解析文件（生成中间 MD）
    new_docs = parse_file_to_md(file_path)
    if not new_docs:
        return "FAILED"

    # 修复核心 Bug：原代码缺少落盘逻辑
    mark_as_processed(file_path) # 登记记录
    rebuild_index_from_md()      # 触发全局极速重建保证FAISS和BM25双路同步
    return "SUCCESS"


# batch_ingest.py (修改 delete_single_file 函数)
def delete_single_file(md5_key):
    """删除指定的文档记录和物理文件，但保留 MD 备份"""
    if not os.path.exists(Config.PROCESSED_RECORD_FILE): return False
    
    with open(Config.PROCESSED_RECORD_FILE, "r", encoding="utf-8") as f:
        records = json.load(f)
        
    if md5_key not in records: return False
    file_path = records[md5_key]
    
    # 1. 从记录字典中删除
    del records[md5_key]
    with open(Config.PROCESSED_RECORD_FILE, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
        
    # 2. 删除原始文件
    if os.path.exists(file_path):
        os.remove(file_path)
        
    # 🚨 修复 2：不再删除对应的中间 MD 文件，将其保留
    # base_name = os.path.splitext(os.path.basename(file_path))[0]
    # md_path = os.path.join("01_RAG", "data", "debug_md", f"{base_name}.md")
    # if os.path.exists(md_path):
    #     os.remove(md_path)
        
    return True

def rebuild_index_from_md():
    """极速重建索引：跳过 PDF 解析，直接读取保存好的 MD 文件算向量"""
    from langchain_community.document_loaders import TextLoader
    
    md_dir = os.path.join("data", "debug_md")
    if not os.path.exists(md_dir): return
    
    all_docs = []
    for filename in os.listdir(md_dir):
        if filename.endswith(".md"):
            md_path = os.path.join(md_dir, filename)
            loader = TextLoader(md_path, encoding="utf-8")
            # 重新加载 MD 内容并附加来源元数据
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = filename.replace(".md", "")
            all_docs.extend(docs)
            
    # 如果把文件删光了，直接清理数据库文件夹
    if not all_docs:
        if os.path.exists(Config.DB_DIR):
            shutil.rmtree(Config.DB_DIR)
        return

    # 直接进入切分和入库阶段
    text_splitter = MarkdownTextSplitter(chunk_size=Config.CHUNK_SIZE, chunk_overlap=Config.CHUNK_OVERLAP)
    chunks = text_splitter.split_documents(all_docs)
    
    if os.path.exists(Config.DB_DIR):
        shutil.rmtree(Config.DB_DIR) # 先清空旧库
    os.makedirs(Config.DB_DIR)
    
    # 极速重建 BM25 和 FAISS
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = Config.RETRIEVER_TOP_K
    with open(os.path.join(Config.DB_DIR, "bm25_index.pkl"), "wb") as f:
        pickle.dump(bm25_retriever, f)
        
    embeddings = OpenAIEmbeddings(
        model=Config.EMBEDDING_MODEL,
        api_key=Config.INTERNAL_API_KEY,
        base_url=Config.INTERNAL_BASE_URL,
        check_embedding_ctx_length=False 
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(Config.DB_DIR)



if __name__ == "__main__":
    target_folder = "data" # 💡 修复：修正 \b 转义 bug
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    else:
        batch_ingest_folder(target_folder)
