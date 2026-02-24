import os
from langchain_openai import ChatOpenAI

# 初始化时自动从环境变量读取 DEEPSEEK_API_KEY（需提前导出）
api_key = os.getenv("DEEPSEEK_API_KEY")
if not api_key:
    raise ValueError("❌ 请设置环境变量 DEEPSEEK_API_KEY")

llm = ChatOpenAI(
    model="deepseek-chat",
    openai_api_key=api_key,
    openai_api_base="https://api.deepseek.com/v1",
)
messages = [
    ("system", "你是一个资深的技术HR，在AI领域经验丰富，非常善于面试agent人才"),
    ("human", "你会怎么挑选AIagent工程师，并附上各能力权重表"),
]
print(llm.invoke(messages))