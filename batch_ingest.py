import os # 导入系统库用于遍历文件夹
import pickle # 导入 pickle 库用于序列化保存 BM25 检索器
from langchain_text_splitters import MarkdownTextSplitter # 导入针对 MD 专门设计的切分器
from langchain_openai import OpenAIEmbeddings # 导入 Embedding 调用类
from langchain_community.vectorstores import FAISS # 导入 FAISS 本地向量库
from langchain_community.retrievers import BM25Retriever # 导入传统的关键词检索器
from file_processor import parse_file_to_md # 导入刚刚写好的路由解析器
from config import Config # 导入全局配置


def batch_ingest_folder(folder_path: str):
    print(f"=== 开始扫描文件夹进行批量入库: {folder_path} ===") # 建议商用取消
    # 初始化一个空列表，用来收集所有文件产生的文档块
    all_docs = []
    
    # 遍历目标文件夹下的所有文件
    for filename in os.listdir(folder_path):
        # 拼接文件的完整路径
        file_path = os.path.join(folder_path, filename)
        # 判断如果是普通文件，则送入处理
        if os.path.isfile(file_path):
            print(f"准备处理文件: {file_path}") # 建议商用取消
            # 核心：调用路由解析器，包含自动查重和 MD 转换
            file_docs = parse_file_to_md(file_path)
            # 如果返回了内容，就把内容追加到全局列表中
            if file_docs:
                all_docs.extend(file_docs)
                
    # 如果扫描完毕发现没有任何新文档产生（可能全是重复的，或文件夹为空），直接退出
    if not all_docs:
        print("没有新增的文档需要入库。流程结束。") # 建议商用取消
        return

    # 初始化 Markdown 切分器，依据 MD 的语法特征进行高级切分
    print("开始进行 Markdown 结构化切分...") # 建议商用取消
    text_splitter = MarkdownTextSplitter(
        chunk_size=Config.CHUNK_SIZE, 
        chunk_overlap=Config.CHUNK_OVERLAP
    )
    # 批量切分前面收集到的所有文档块
    chunks = text_splitter.split_documents(all_docs)
    print(f"共生成 {len(chunks)} 个切分后的 Chunk") # 建议商用取消
    
    # 确保保存数据库的文件夹已经创建
    if not os.path.exists(Config.DB_DIR):
        os.makedirs(Config.DB_DIR)

    # ==== 开始构建 BM25 关键词索引 ====
    print("正在构建并固化 BM25 关键词索引...") # 建议商用取消
    # 使用所有切分后的块生成 BM25 对象
    bm25_retriever = BM25Retriever.from_documents(chunks)
    # 设置关键词检索的截断数量
    bm25_retriever.k = Config.RETRIEVER_TOP_K
    # 以二进制写模式序列化并保存 BM25 对象到本地
    with open(os.path.join(Config.DB_DIR, "bm25_index.pkl"), "wb") as f:
        pickle.dump(bm25_retriever, f)

    # ==== 开始构建 FAISS 向量数据库 ====
    print("正在调用内网 Embedding 接口计算向量...") # 建议商用取消
    # 初始化 Embedding 客户端，使用全局统一定义的接口地址和密钥
    embeddings = OpenAIEmbeddings(
        model=Config.EMBEDDING_MODEL,
        api_key=Config.INTERNAL_API_KEY,
        base_url=Config.INTERNAL_BASE_URL,
        check_embedding_ctx_length=False 
    )
    # 针对所有 chunk 批量请求向量，并构建 FAISS 索引
    vectorstore = FAISS.from_documents(chunks, embeddings)
    # FAISS 自带落盘方法，直接存入指定的本地文件夹
    vectorstore.save_local(Config.DB_DIR)
    
    # 打印入库成功提示
    print(f"=== 批量入库完成！数据已安全落盘至 {Config.DB_DIR}/ 目录 ===") # 建议商用取消

if __name__ == "__main__":
    # 假设你的待入库文件都放在 data 文件夹下
    target_folder = "01_RAG\data" 
    # 如果文件夹不存在，自动创建
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
        print(f"已创建 {target_folder} 文件夹，请放入文件后重试。") # 建议商用取消
    else:
        # 执行批量入库主流程
        batch_ingest_folder(target_folder)
