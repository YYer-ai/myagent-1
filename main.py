import os
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.tools import tool,ToolRuntime
from langchain.messages import HumanMessage, AIMessage, SystemMessage
from langchain.agents.middleware import dynamic_prompt, ModelRequest
from langgraph.types import Command
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# 初始化时自动从环境变量读取 DEEPSEEK_API_KEY（需提前导出）
api_key = os.getenv("DEEPSEEK_API_KEY")
if not api_key:
    raise ValueError("❌ 请设置环境变量 DEEPSEEK_API_KEY")

deepseek = ChatOpenAI(
    model="deepseek-chat",
    openai_api_key=api_key,
    openai_api_base="https://api.deepseek.com/v1",
)

agent = create_agent(
    model=deepseek,
    name="YY",
    )

# 1. 准备存储容器（用于存放不同 Session 的对话历史）
store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

def chat_with_agent():
    # 2. 定义 Prompt，关键在于 MessagesPlaceholder
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个精通诗词的专家，名字是iYY。"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ])

    # 3. 构建 Chain（假设你的 agent 已经定义好）
    # 注意：如果你的 agent 是 LangGraph 创建的，请确保它接收 messages 序列
    chain = prompt | agent

    # 4. 包装成带历史记录的 Runnable
    with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )

    # 5. 进入对话循环
    session_config = {"configurable": {"session_id": "user_001"}}
    
    while True:
        user_input = input("你: ")
        if user_input.lower() in ['退出', 'exit', 'quit']:
            print("撒由那拉，再会喽！")
            break

        # invoke 时，它会自动去 store 里找 history 并注入
        response = with_history.invoke(
            {"input": user_input},
            config=session_config
        )

        # 处理 Agent 返回字典的情况
        if isinstance(response, dict) and 'messages' in response:
            content = response['messages'][-1].content
        else:
            content = response.content

        print(f"iYY: {content}")

# 启动
chat_with_agent()