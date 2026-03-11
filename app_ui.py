import streamlit as st
import os
from batch_ingest import batch_ingest_folder
from query_service import build_query_chain

st.set_page_config(page_title="企业级智能 RAG 助手", layout="wide")
# 1. 显示已入库文件清单
def get_ingested_files():
    if os.path.exists(Config.PROCESSED_RECORD_FILE):
        with open(Config.PROCESSED_RECORD_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

ingested_data = get_ingested_files()

with st.sidebar:
    st.header("📂 已入库文件清单")
    if ingested_data:
        for md5, path in ingested_data.items():
            st.text(f"📄 {os.path.basename(path)}")
    else:
        st.info("暂无入库文件")

# 2. 上传与覆盖逻辑
uploaded_file = st.file_uploader("选择文件上传", type=['pdf', 'docx', 'jpg', 'png'])

if uploaded_file:
    file_name = uploaded_file.name
    temp_path = os.path.join("01_RAG/data", file_name)
    
    # 模拟保存
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # 检查是否存在
    is_exist = any(file_name in p for p in ingested_data.values())
    
    if is_exist:
        st.warning(f"检测到文件 `{file_name}` 已存在。")
        col1, col2 = st.columns(2)
        if col1.button("覆盖导入"):
            status = ingest_single_file(temp_path, force_overwrite=True)
            st.success("覆盖成功！")
        if col2.button("取消"):
            st.info("操作已取消")
    else:
        if st.button("确认入库"):
            with st.spinner("解析中..."):
                status = ingest_single_file(temp_path)
                st.success("入库成功！")

# --- 侧边栏：管理知识库 ---
with st.sidebar:
    st.title("📚 知识库管理")
    uploaded_files = st.file_uploader("上传新文件 (PDF/Docx/JPG)", accept_multiple_files=True)
    
    if st.button("🚀 开始同步入库"):
        if uploaded_files:
            # 保存到临时目录
            save_path = "01_RAG/data"
            os.makedirs(save_path, exist_ok=True)
            for f in uploaded_files:
                with open(os.path.join(save_path, f.name), "wb") as buffer:
                    buffer.write(f.read())
            
            with st.spinner("正在解析、转换并构建索引..."):
                batch_ingest_folder(save_path) # 调用你已有的入库逻辑
            st.success("入库成功！")
        else:
            st.warning("请先选择文件")

# --- 主界面：对话窗口 ---
st.title("🤖 智能问答终端")

if "messages" not in st.session_state:
    st.session_state.messages = []

# 显示历史聊天记录
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 初始化 RAG 链 (Session 缓存避免重复加载)
if "rag_chain" not in st.session_state:
    try:
        st.session_state.rag_chain = build_query_chain() # 调用你已有的查询链
    except:
        st.info("请先在左侧上传并入库文件以激活助手")

# 聊天输入
if prompt := st.chat_input("关于文档你想了解什么？"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if "rag_chain" in st.session_state:
            with st.spinner("正在检索实时资料并思考..."):
                # 执行 RAG 逻辑
                response = st.session_state.rag_chain.invoke(prompt)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
        else:
            st.error("知识库未就绪，请先完成左侧入库步骤。")
