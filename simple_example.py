"""
LangChain DeepSeek 最小化示例
只使用环境变量设置 API key
"""

from langchain_deepseek import ChatDeepSeek
import os

# 方法1: 在运行脚本前设置环境变量
# Windows: set DEEPSEEK_API_KEY=your-api-key
# Linux/Mac: export DEEPSEEK_API_KEY='your-api-key'

# 方法2: 在代码中设置环境变量（不推荐用于生产环境）
# os.environ['DEEPSEEK_API_KEY'] = 'your-api-key'

# 检查环境变量
api_key = os.getenv("DEEPSEEK_API_KEY")
if not api_key:
    print("错误: 请设置 DEEPSEEK_API_KEY 环境变量")
    print("设置方法:")
    print("  Windows: set DEEPSEEK_API_KEY=your-api-key")
    print("  Linux/Mac: export DEEPSEEK_API_KEY='your-api-key'")
    exit(1)

print(f"使用 API key: {api_key[:8]}...")

# 创建模型实例（不传递 api_key 参数，从环境变量读取）
model = ChatDeepSeek(
    model="deepseek-chat",
    temperature=0.7,
    max_tokens=100
)

# 定义对话
messages = [
    ("system", "你是一个有用的助手。"),
    ("human", "请用一句话介绍 LangChain。"),
]

# 调用模型
try:
    response = model.invoke(messages)
    print(f"\n回答: {response.content}")
except Exception as e:
    print(f"\n错误: {e}")
