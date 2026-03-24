# -*- coding: utf-8 -*-
"""
测试压缩会话历史管理功能
"""

import os
import sys
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from sqlalchemy import create_engine
from compressed_history import get_compressed_session_history

# 设置控制台输出编码
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# 初始化 LLM
api_key = os.getenv("DEEPSEEK_API_KEY")
if not api_key:
    raise ValueError("请设置环境变量 DEEPSEEK_API_KEY")

deepseek = ChatOpenAI(
    model="deepseek-chat",
    openai_api_key=api_key,
    openai_api_base="https://api.deepseek.com/v1",
)

# 初始化数据库引擎
engine = create_engine("sqlite:///test_chat_history.db")

# 测试参数
test_session_id = "test_session_001"
buffer_size = 5  # 保留 5 条原始消息
compress_threshold = 8  # 超过 8 条时触发压缩
compress_ratio = 0.5  # 压缩最旧的 50%

print("[测试] 开始测试压缩会话历史管理\n")
print(f"[配置] 参数:")
print(f"   - 缓冲区大小: {buffer_size} 条")
print(f"   - 压缩阈值: {compress_threshold} 条")
print(f"   - 压缩比例: {compress_ratio * 100}%\n")

# 创建会话历史
history = get_compressed_session_history(
    session_id=test_session_id,
    engine=engine,
    llm_for_compression=deepseek,
    buffer_size=buffer_size,
    compress_threshold=compress_threshold,
    compress_ratio=compress_ratio,
)

# 添加测试消息
test_messages = [
    HumanMessage(content="你好，我想了解一下唐诗"),
    AIMessage(content="唐诗是中国古典诗歌的巅峰之作，代表人物有李白、杜甫等。"),
    HumanMessage(content="李白的《静夜思》是什么？"),
    AIMessage(content="《静夜思》是李白的代表作之一：床前明月光，疑是地上霜。举头望明月，低头思故乡。"),
    HumanMessage(content="这首诗表达了什么情感？"),
    AIMessage(content="这首诗表达了诗人对故乡的深切思念，以及身在异乡的孤独感。"),
    HumanMessage(content="杜甫的诗风有什么特点？"),
    AIMessage(content="杜甫的诗风沉郁顿挫，关注现实，被称为诗圣，与李白并称李杜。"),
    HumanMessage(content="他们两人的区别是什么？"),
    AIMessage(content="李白浪漫主义，豪放飘逸；杜甫现实主义，忧国忧民。两人风格迥异但各有千秋。"),
]

print("[添加] 测试消息...")
for i, msg in enumerate(test_messages, 1):
    history.add_message(msg)
    print(f"   {i}. [{msg.type}] {msg.content[:30]}...")

print(f"\n[统计] 当前消息总数: {len(history.messages)}")

# 显示当前消息列表
print("\n[历史] 当前会话历史:")
for i, msg in enumerate(history.messages, 1):
    prefix = "[压缩]" if "摘要" in msg.content else "[原始]"
    print(f"   {prefix} {i}. [{msg.type}] {msg.content}")

# 测试持久化 - 重新加载会话历史
print("\n[持久化] 测试 - 重新加载会话历史...")
history_reloaded = get_compressed_session_history(
    session_id=test_session_id,
    engine=engine,
    llm_for_compression=deepseek,
    buffer_size=buffer_size,
    compress_threshold=compress_threshold,
    compress_ratio=compress_ratio,
)

print(f"[统计] 重新加载后的消息总数: {len(history_reloaded.messages)}")
print("\n[历史] 重新加载后的会话历史:")
for i, msg in enumerate(history_reloaded.messages, 1):
    prefix = "[压缩]" if "摘要" in msg.content else "[原始]"
    print(f"   {prefix} {i}. [{msg.type}] {msg.content}")

# 清理
print("\n[清理] 测试数据...")
history_reloaded.clear()
print("[完成] 测试成功！")
