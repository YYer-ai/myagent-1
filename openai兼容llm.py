from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="deepseek-chat",          # 或其他 DeepSeek 模型名
    openai_api_key= # 或通过环境变量设置
    openai_api_base="https://api.deepseek.com/v1",
    temperature=0.7,
)
messages = [
    ("system", "You are a helpful translator. Translate the user sentence to French."),
    ("human", "I love programming."),
]
print(llm.invoke(messages))
