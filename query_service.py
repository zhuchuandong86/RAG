import os # 导入操作系统库
import pickle # 导入 pickle 库用于加载之前序列化的 BM25
from langchain_openai import OpenAIEmbeddings, ChatOpenAI # 导入所需大模型调用模块
from langchain_community.vectorstores import FAISS # 导入 FAISS 模块进行向量检索
from langchain_core.prompts import ChatPromptTemplate # 导入模板格式化模块
from langchain_core.output_parsers import StrOutputParser # 导入结果字符解析模块
from langchain_core.runnables import RunnablePassthrough # 导入 LCEL 语法透传组件
from langchain_classic.retrievers import EnsembleRetriever # 导入双路混合检索器组件

from config import Config # 导入全局配置
# 假设你之前写的基于内网 API 的 reranker 放在 reranker.py 里，这里导入构建函数
from reranker import build_rerank_retriever 



def format_docs(docs):
    # 调试：打印出到底召回了哪些片段
    for i, doc in enumerate(docs):
        print(f"--- 召回片段 {i+1} (来源: {doc.metadata.get('page', '未知')}页) ---")
        print(f"{doc.page_content[:200]}...") 
    return "\n\n".join(doc.page_content for doc in docs)
    

def build_query_chain():
    print("=== 系统启动中：尝试从硬盘加载知识库 ===") # 建议商用取消
    # 校验本地是否存在数据库文件夹
    if not os.path.exists(Config.DB_DIR):
        raise FileNotFoundError(f"找不到数据库 {Config.DB_DIR}，请先运行 batch_ingest.py 入库！")
        
    # 1. 从硬盘读取 BM25 检索器对象
    print("1. 正在反序列化加载 BM25...") # 建议商用取消
    with open(os.path.join(Config.DB_DIR, "bm25_index.pkl"), "rb") as f:
        bm25_retriever = pickle.load(f)
        
    # 2. 重新初始化 Embedding，并加载本地 FAISS 向量库
    print("2. 正在挂载 FAISS 向量库...") # 建议商用取消
    embeddings = OpenAIEmbeddings(
        model=Config.EMBEDDING_MODEL,
        api_key=Config.INTERNAL_API_KEY,
        base_url=Config.INTERNAL_BASE_URL,
        check_embedding_ctx_length=False 
    )
    # 加载本地向量索引（声明已知其安全性）
    vectorstore = FAISS.load_local(Config.DB_DIR, embeddings, allow_dangerous_deserialization=True)
    # 将 FAISS 对象转为 LangChain 检索器模式，限制召回数量
    faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": Config.RETRIEVER_TOP_K})
    
    # 3. 将两路召回结合，组合为 EnsembleRetriever（各占 50% 权重）
    print("3. 组装双路混合检索引擎...") # 建议商用取消
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever],
        weights=[0.5, 0.5]
    )
    
    
    # 4. 在混合检索器外面包一层你自定义的 Reranker 精排通道
    print("4. 挂载 Reranker 精排管道...") # 建议商用取消
    final_retriever = build_rerank_retriever(ensemble_retriever)
    
    # 5. 定义用于指导大模型回答的核心 Prompt 模板
    prompt = ChatPromptTemplate.from_template("""
你是一个专业的智能助理。请严格基于以下提供的参考资料来解答用户的疑问。
参考资料:
{context}

用户问题: {question}
回答:""")
    
    # 6. 初始化内网问答大模型 (统一使用 Config 中配置的纯文本模型)
    print(f"5. 初始化大语言模型 {Config.MODEL_NAME}...") # 建议商用取消
    llm = ChatOpenAI(
        model=Config.MODEL_NAME,
        api_key=Config.INTERNAL_API_KEY,
        base_url=Config.INTERNAL_BASE_URL,
        temperature=0.1 # 问答系统调低随机性，保障回答稳定可靠
    )
    
    # 7. 组装 LCEL 标准大模型流水线
    rag_chain = (
        # 组装输入：将 query 透传给 question 槽，同时将 query 丢给最终检索器，结果交由 format_docs 拼装给 context 槽
        {"context": final_retriever | format_docs, "question": RunnablePassthrough()}
        | prompt           # 将槽位注入到提示词模板
        | llm              # 提示词丢给大语言模型进行推理
        | StrOutputParser() # 解析大语言模型返回的消息结构，纯净提取字符串
    )
    
    print("=== 准备就绪，开启智能查询通道 ===") # 建议商用取消
    return rag_chain

if __name__ == "__main__":
    # 实例化调用整个流水线
    chain = build_query_chain()
    # 用 input 模拟 API 请求或用户发问
    query = input("\n请提出你的问题: ")
    print("思考推理中...") # 建议商用取消
    # 触发 RAG 流水线运行并获取回答结果
    result = chain.invoke(query)
    print(f"\n回答:\n{result}")
