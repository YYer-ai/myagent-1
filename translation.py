import os
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


# 初始化时自动从环境变量读取 DEEPSEEK_API_KEY（需提前导出）
api_key = os.getenv("DEEPSEEK_API_KEY")
if not api_key:
    raise ValueError("❌ 请设置环境变量 DEEPSEEK_API_KEY")

deepseek = ChatOpenAI(
    model="deepseek-chat",
    openai_api_key=api_key,
    openai_api_base="https://api.deepseek.com/v1",
)

prompt = ChatPromptTemplate([
    ("system","把用户输入的中文翻译成{target_language}"),
    ("user","{input_text}"),
])
prompt_1 = prompt.format(target_language="英语", input_text="今天天气怎么样？")
result = deepseek.invoke(prompt_1)
print(result)
print('------------------'*2)
str_parser = StrOutputParser()
str_result = str_parser.invoke(result)
print("stroutputparser:", str_result)
print('------------------'*2)