import base64 # 导入 base64 库，用于图片的编码转换
import requests # 导入 requests 库，用于发起 HTTP 请求
from config import Config # 引入全局配置项

def text_to_md(raw_text: str) -> str:
    # 检查输入文本是否为空，为空则直接返回空字符串
    if not raw_text:
        return ""
    # 将文本中多余的连续三个换行替换为两个换行，并去除首尾空白字符
    clean_text = raw_text.replace('\n\n\n', '\n\n').strip()
    print("已完成纯文本格式的清洗与整理") # 建议商用取消
    # 返回清理后的纯文本，可以直接视作最基础的 Markdown
    return clean_text

def image_to_md_via_vlm(image_path: str) -> str:
    # 尝试执行视觉大模型调用逻辑
    try:
        print(f"准备将图片传入视觉大模型进行 MD 转换: {image_path}") # 建议商用取消
        # 以二进制只读模式打开图片文件
        with open(image_path, "rb") as image_file:
            # 读取图片内容，并将其进行 base64 编码，再解码为 utf-8 字符串格式
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
            
        # 组装请求头，指定数据类型并带上统一的 API 密钥进行鉴权
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {Config.INTERNAL_API_KEY}"
        }
        
        # 组装请求体数据（适配主流兼容 OpenAI 的内网 VLM 接口）
# md_converter.py (修改 image_to_md_via_vlm 函数中的 payload)
        payload = {
            "model": Config.MODEL_VISION, 
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text", 
                            # 🚨 修复 3：强化提示词约束，严禁模型自行编造和输出页码
                            "text": "提取图片中的文字、表格、标题，使用标准 Markdown 格式输出。严格要求：不要废话，绝不要自行编造、添加或输出任何页码信息（如'第x页'），忽略图片边缘的页脚页眉数字。"
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                        }
                    ]
                }
            ],
            "temperature": 0.0 # 将温度调至 0.0，最大程度降低模型幻觉发散的概率
        }
        # 打印正在发起的网络请求信息
        print(f"正在调用 {Config.MODEL_VISION} 接口识别图片...") # 建议商用取消
        # 向统一的内网 API 地址发送 POST 请求，如果你内网聊天和 VLM 地址有细微区别，请微调 URL 路径后缀
        response = requests.post(f"{Config.INTERNAL_BASE_URL}/chat/completions", headers=headers, json=payload)
        # 如果 HTTP 状态码不是 200，则主动抛出异常
        response.raise_for_status()
        # 将返回的 JSON 响应数据解析为字典
        result = response.json()
        # 提取模型回复的内容，去除首尾空格
        md_content = result["choices"][0]["message"]["content"].strip()
        print("图片已成功转换为 Markdown 格式") # 建议商用取消
        # 返回最终的 MD 文本
        return md_content
        
    # 捕获所有运行过程中的异常
    except Exception as e:
        # 打印异常信息
        print(f"视觉解析失败 ({image_path}): {e}") # 建议商用取消
        # 失败则返回空字符串，避免阻断整个流水线
        return ""
