import os
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.tools import tool,ToolRuntime
from langchain.messages import HumanMessage, AIMessage, SystemMessage
from typing import TypedDict
from langchain.agents.middleware import dynamic_prompt, ModelRequest

# 初始化时自动从环境变量读取 DEEPSEEK_API_KEY（需提前导出）
api_key = os.getenv("DEEPSEEK_API_KEY")
if not api_key:
    raise ValueError("❌ 请设置环境变量 DEEPSEEK_API_KEY")

deepseek = ChatOpenAI(
    model="deepseek-chat",
    openai_api_key=api_key,
    openai_api_base="https://api.deepseek.com/v1",
)


@tool
def get_last_user_message(runtime: ToolRuntime) -> str:
    """Get the most recent message from the user."""
    messages = runtime.state["messages"]

    # Find the last human message
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return message.content

    return "No user messages found"

# Access custom state fields
@tool
def get_user_preference(
    pref_name: str,
    runtime: ToolRuntime
) -> str:
    """Get a user preference value."""
    preferences = runtime.state.get("user_preferences", {})
    return preferences.get(pref_name, "Not set")

@tool
def search_database(query: str, limit: int = 10) -> str:
    """Search the customer database for records matching the query.

    Args:
        query: Search terms to look for
        limit: Maximum number of results to return
    """
    return f"Found {limit} results for '{query}'"

class Context(TypedDict):
    user_role: str

@dynamic_prompt
def user_role_prompt(request: ModelRequest) -> str:
    """Generate system prompt based on user role."""
    user_role = request.runtime.context.get("user_role", "user")
    base_prompt = "你是一个智能助手，能够根据用户角色调整回答的详细程度。"

    if user_role == "专家":
        return f"{base_prompt} 提供详细科学的解释，并使用专业术语。"
    elif user_role == "初学者":
        return f"{base_prompt} 解释尽量通俗易懂，避免专业术语。"

    return base_prompt

agent = create_agent(
    model=deepseek,
    name="YY",
    middleware=[user_role_prompt],
    tools=[search_database, get_last_user_message, get_user_preference]
    )

Human_msg = HumanMessage(content="你叫什么，你是什么？")                        
# agent.invoke() 应该接收一个字典，包含输入消息和上下文信息
input_dict={
    "messages": [Human_msg],
}
response = agent.invoke(input_dict, context={"user_role": "expert"})
print(response)
