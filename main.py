import os
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

# 初始化时自动从环境变量读取 DEEPSEEK_API_KEY（需提前导出）
api_key = os.getenv("DEEPSEEK_API_KEY")
if not api_key:
    raise ValueError("❌ 请设置环境变量 DEEPSEEK_API_KEY")

deepseek = ChatOpenAI(
    model="deepseek-chat",
    openai_api_key=api_key,
    openai_api_base="https://api.deepseek.com/v1",
)

def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

agent = create_agent(
    model=deepseek,
    system_prompt="你是一个资深的技术HR，在AI领域经验丰富，非常善于面试agent人才",
    tools=[get_weather])
    
messages = [
    {"role": "user", "content": "你会怎么挑选AIagent工程师，并附上各能力权重表"}
    ]

input_dict = {"messages": messages} 
response = agent.invoke(input_dict)

# 提取AI响应内容
ai_content = None

# 从响应中提取AI消息内容
if isinstance(response, dict) and 'messages' in response:
    for msg in response['messages']:
        if hasattr(msg, 'content') and 'AIMessage' in str(type(msg)):
            ai_content = msg.content
            break

# 打印AI响应
print("=" * 60)
print("AI Agent 响应:")
print("=" * 60)

if ai_content:
    try:
        # 尝试直接打印
        print(ai_content)
    except UnicodeEncodeError:
        try:
            # 尝试使用GBK编码处理中文字符
            print(ai_content.encode('gbk', errors='replace').decode('gbk'))
        except:
            # 最后使用repr
            print(f"响应内容: {repr(ai_content)}")
else:
    # 如果没有找到AI内容，打印整个响应
    try:
        print(str(response))
    except UnicodeEncodeError:
        print(repr(response))

print("=" * 60)
