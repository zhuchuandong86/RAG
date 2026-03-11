import streamlit as st
import os
import json
from config import Config
from batch_ingest import ingest_single_file, delete_single_file, rebuild_index_from_md
from query_service import build_query_chain

st.set_page_config(page_title="RAG 智库商用版", layout="wide")

# 🚨 核心修复 1：使用官方缓存机制，防止重复加载和双重打印
@st.cache_resource
def get_rag_chain():
    try:
        return build_query_chain()
    except Exception as e:
        return None

# 获取模型链实例
rag_chain = get_rag_chain()

def get_ingested_files():
    if os.path.exists(Config.PROCESSED_RECORD_FILE):
        with open(Config.PROCESSED_RECORD_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

# 辅助函数：强制重载知识库
def reload_knowledge_base():
    if "rag_chain" in st.session_state:
        del st.session_state["rag_chain"]

# --- 侧边栏：管理与清单 ---
with st.sidebar:
    st.title("📚 知识库中心")
    
    st.subheader("📂 已存文档")
    ingested_data = get_ingested_files()
    if ingested_data:
        for md5_key, path in ingested_data.items():
            # 修复 1：去掉 caption，改用 columns 和 markdown 实现正常字体与删除按钮
            col1, col2 = st.columns([4, 1])
            col1.markdown(f"📄 **{os.path.basename(path)}**")
            
            # 修复 2：增加删除功能
            if col2.button("❌", key=f"del_{md5_key}", help="删除此文档"):
                with st.spinner("正在删除文档并极速重建索引..."):
                    delete_single_file(md5_key)
                    rebuild_index_from_md() # 重建底层库
                    reload_knowledge_base() # 重置大模型连接
                st.success("删除成功！")
                st.rerun()
    else:
        st.info("暂无数据")
    
    st.divider()

    st.subheader("📤 上传并入库")
    uploaded_file = st.file_uploader("选择文档", type=['pdf', 'docx', 'png', 'jpg'])
    
    if uploaded_file:
        temp_dir = "data"
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, uploaded_file.name)
        
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        is_exist = any(uploaded_file.name in p for p in ingested_data.values())
        
        if is_exist:
            st.warning(f"文件 `{uploaded_file.name}` 已存在。")
            if st.button("🔥 覆盖导入"):
                with st.spinner("重构索引中..."):
                    status = ingest_single_file(temp_path, force_overwrite=True)
                    if status == "SUCCESS":
                        reload_knowledge_base()
                        st.success("覆盖完成！")
                    else:
                        st.error("解析失败：文件可能损坏或内容无法提取。")
                st.rerun()
        else:
            if st.button("✅ 确认入库"):
                with st.spinner("解析并算向量中..."):
                    status = ingest_single_file(temp_path)
                    if status == "SUCCESS":
                        reload_knowledge_base()
                        st.success("入库成功！")
                    else:
                        st.error("解析失败：文件未入库，请检查文件完整性。")
                st.rerun()

# --- 主界面：问答与引用高亮 ---
st.title("🤖 智能对话终端")

if "messages" not in st.session_state:
    st.session_state.messages = []

# 初始化 RAG 链 (如果被 reload_knowledge_base 删除了，这里会重新建立连接)
if "rag_chain" not in st.session_state:
    try:
        st.session_state.rag_chain = build_query_chain()
    except:
        st.info("💡 请先在左侧入库文档以激活搜索。")

# 显示历史
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg:
            with st.expander("🔍 原始引用片段"):
                for doc in msg["sources"]:
                    st.caption(f"来源：{os.path.basename(doc.metadata.get('source', '未知'))}")
                    st.code(doc.page_content, language="markdown")

# 问答交互
if prompt := st.chat_input("关于您的文档，想问点什么？"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if rag_chain is not None:  # 改为直接判断对象是否存在
            with st.spinner("正在检索实时数据并推理..."):
                retriever = rag_chain.first["context"]
                source_docs = retriever.invoke(prompt)
                
                res = rag_chain.invoke(prompt)
                st.markdown(res)
                
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": res, 
                    "sources": source_docs
                })
        else:
            st.error("知识库未就绪。请上传文件。")
