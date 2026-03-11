# app_ui.py
import streamlit as st
import os
import json
from config import Config
from file_processor import get_file_md5
from batch_ingest import ingest_single_file, delete_single_file, rebuild_index_from_md
from query_service import build_query_chain
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

st.set_page_config(page_title="RAG 智库商用版", layout="wide")

# 🚨 修复 1：缓存函数支持接收两个返回值
@st.cache_resource
def get_rag_chain_and_retriever():
    try:
        return build_query_chain()
    except Exception as e:
        return None, None

# 获取模型链和检索器实例
rag_chain, retriever = get_rag_chain_and_retriever()
# AI 智能命名函数
def generate_ai_filename(original_name):
    try:
        llm = ChatOpenAI(model=Config.MODEL_NAME, api_key=Config.INTERNAL_API_KEY, base_url=Config.INTERNAL_BASE_URL, temperature=0.7)
        prompt = f"请根据原文件名 '{original_name}'，生成一个简短且规范的中文标题名（只输出名字本身，不要带扩展名，不要任何解释）。"
        new_name = llm.invoke(prompt).content.strip()
        ext = os.path.splitext(original_name)[-1]
        return new_name + ext
    except:
        return original_name

# 读取真正的向量库验证入库清单
def get_ingested_files():
    if os.path.exists(Config.DB_DIR) and os.path.exists(os.path.join(Config.DB_DIR, "index.faiss")):
        try:
            embeddings = OpenAIEmbeddings(
                model=Config.EMBEDDING_MODEL, api_key=Config.INTERNAL_API_KEY, 
                base_url=Config.INTERNAL_BASE_URL, check_embedding_ctx_length=False 
            )
            # 加载向量库解析元数据
            vectorstore = FAISS.load_local(Config.DB_DIR, embeddings, allow_dangerous_deserialization=True)
            sources = {}
            for doc in vectorstore.docstore._dict.values():
                src = doc.metadata.get("source")
                if src and src not in sources.values():
                    # 重新生成 key，保持与此前 JSON 一致的操作逻辑
                    md5_key = get_file_md5(src) if os.path.exists(src) else src
                    sources[md5_key] = src
            return sources
        except Exception as e:
            pass
    
    # 若库不存在则降级读取日志文件
    if os.path.exists(Config.PROCESSED_RECORD_FILE):
        with open(Config.PROCESSED_RECORD_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def reload_knowledge_base():
    if "rag_chain" in st.session_state:
        del st.session_state["rag_chain"]

# --- 侧边栏 ---
with st.sidebar:
    st.title("📚 知识库中心")
    st.subheader("📂 已存文档")
    ingested_data = get_ingested_files()
    
    if ingested_data:
        for md5_key, path in ingested_data.items():
            col1, col2 = st.columns([4, 1])
            col1.markdown(f"📄 **{os.path.basename(path)}**")
            if col2.button("❌", key=f"del_{md5_key}"):
                with st.spinner("删除记录中（保留 MD 备份）..."):
                    delete_single_file(md5_key)
                    # 🚨 修复 2：不再自动调用 rebuild_index_from_md()
                    reload_knowledge_base()
                st.success("删除成功！重新入库请手动上传备份的 MD 文件。")
                st.rerun()
    else:
        st.info("暂无数据")
    
    st.divider()
    st.subheader("📤 上传并入库")
    
    # 扩充支持的格式
    uploaded_file = st.file_uploader("选择文档", type=['pdf', 'docx', 'doc', 'png', 'jpg', 'txt', 'md'])
    
    if uploaded_file:
        temp_dir = "data"
        os.makedirs(temp_dir, exist_ok=True)
        
        # 触发智能命名
        with st.spinner("AI 正在根据文件名智能生成标题..."):
            smart_filename = generate_ai_filename(uploaded_file.name)
            
        temp_path = os.path.join(temp_dir, smart_filename)
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # 根据智能命名后的结果查重
        is_exist = any(smart_filename in p for p in ingested_data.values())
        
        if is_exist:
            st.warning(f"文件 `{smart_filename}` 已存在。")
            if st.button("🔥 覆盖导入"):
                with st.spinner("重构索引中..."):
                    status = ingest_single_file(temp_path, force_overwrite=True)
                    if status == "SUCCESS":
                        reload_knowledge_base()
                        st.success("覆盖完成！")
                        st.rerun() # 入库完成，立即刷新左上角已存文档列表
                    else:
                        st.error("解析失败。")
        else:
            if st.button(f"✅ 确认入库 ({smart_filename})"):
                with st.spinner("解析并算向量中..."):
                    status = ingest_single_file(temp_path)
                    if status == "SUCCESS":
                        reload_knowledge_base()
                        st.success("入库成功！")
                        st.rerun() # 触发全局刷新同步最新数据库数据
                    else:
                        st.error("入库失败。")

# --- 主界面 ---
st.title("🤖 智能对话终端")

if "messages" not in st.session_state:
    st.session_state.messages = []

# 🚨 修复 2：初始化时接收两个变量存入 session_state
if "rag_chain" not in st.session_state:
    try:
        st.session_state.rag_chain, st.session_state.retriever = build_query_chain()
    except:
        st.info("💡 请先在左侧入库文档以激活搜索。")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg:
            with st.expander("🔍 原始引用片段"):
                for doc in msg["sources"]:
                    # 前端显式呈现文件的具体页码
                    page_info = f" - 第{doc.metadata.get('page')}页" if 'page' in doc.metadata else ""
                    st.caption(f"来源：{os.path.basename(doc.metadata.get('source', '未知'))}{page_info}")
                    st.code(doc.page_content, language="markdown")

if prompt := st.chat_input("关于您的文档，想问点什么？"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if "rag_chain" in st.session_state and st.session_state.rag_chain is not None:  
            with st.spinner("检索中..."):
                # 🚨 修复 3：直接调用我们独立拿出来的 retriever，不再去解剖 rag_chain
                source_docs = st.session_state.retriever.invoke(prompt)
                
                res = st.session_state.rag_chain.invoke(prompt)
                st.markdown(res)
                
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": res, 
                    "sources": source_docs
                })
        else:
            st.error("知识库未就绪。请上传文件。")
