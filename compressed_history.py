"""
压缩式会话历史管理模块

实现智能的会话历史管理策略：
- 保留最近 n 条原始对话
- 对超出缓冲区的旧对话进行摘要压缩
- 持久化存储到 SQLite
"""

import json
from typing import List, Optional, Dict, Any
from langchain_core.messages import BaseMessage, messages_from_dict, messages_to_dict
from langchain_core.chat_history import BaseChatMessageHistory
from sqlalchemy import create_engine, Column, String, Integer, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os

Base = declarative_base()


class CompressedMessage(Base):
    """压缩消息存储表"""
    __tablename__ = "compressed_messages"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, index=True)
    message_type = Column(String)  # 'summary' 或 'raw'
    content = Column(Text)  # JSON 格式的消息内容
    timestamp = Column(String)  # ISO 格式时间戳
    order_index = Column(Integer)  # 消息顺序索引


class CompressedChatHistory(BaseChatMessageHistory):
    """
    压缩式会话历史管理类

    特性：
    - 保留最近 n 条原始对话
    - 对旧对话进行摘要压缩
    - 自动压缩和持久化
    """

    def __init__(
        self,
        session_id: str,
        engine,
        llm_for_compression,  # 用于压缩的 LLM
        buffer_size: int = 10,  # 保留的原始对话数量
        compress_threshold: int = 15,  # 触发压缩的总消息数阈值
        compress_ratio: float = 0.5,  # 压缩比例（保留旧消息的比例）
    ):
        """初始化压缩会话历史

        Args:
            session_id: 会话ID
            engine: SQLAlchemy 数据库引擎
            llm_for_compression: 用于消息压缩的 LLM 模型
            buffer_size: 保留的原始对话数量（轮次）
            compress_threshold: 触发压缩的总消息数阈值
            compress_ratio: 压缩比例（旧消息中保留的比例）
        """
        self.session_id = session_id
        self.engine = engine
        self.llm_for_compression = llm_for_compression
        self.buffer_size = buffer_size
        self.compress_threshold = compress_threshold
        self.compress_ratio = compress_ratio

        # 创建表（如果不存在）
        Base.metadata.create_all(engine)

        # 从数据库加载消息
        self._load_messages()

    def _load_messages(self):
        """从数据库加载所有消息"""
        Session = sessionmaker(bind=self.engine)
        session = Session()

        try:
            # 按 order_index 加载消息
            records = session.query(CompressedMessage).filter(
                CompressedMessage.session_id == self.session_id
            ).order_by(CompressedMessage.order_index).all()

            self.messages: List[BaseMessage] = []
            for record in records:
                msg_dict = json.loads(record.content)
                msg = messages_from_dict([msg_dict])[0]
                self.messages.append(msg)

        finally:
            session.close()

    def _save_messages(self):
        """保存所有消息到数据库"""
        Session = sessionmaker(bind=self.engine)
        session = Session()

        try:
            # 删除旧记录
            session.query(CompressedMessage).filter(
                CompressedMessage.session_id == self.session_id
            ).delete()

            # 插入新记录
            msg_dicts = messages_to_dict(self.messages)
            for idx, msg_dict in enumerate(msg_dicts):
                record = CompressedMessage(
                    session_id=self.session_id,
                    message_type="raw",
                    content=json.dumps(msg_dict, ensure_ascii=False),
                    timestamp=datetime.now().isoformat(),
                    order_index=idx
                )
                session.add(record)

            session.commit()
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()

    def _compress_old_messages(self):
        """压缩旧消息

        策略：
        1. 保留最新的 buffer_size 条消息
        2. 将最旧的 compress_ratio 部分消息压缩成摘要
        """
        if len(self.messages) < self.compress_threshold:
            return

        # 计算压缩的边界
        total_messages = len(self.messages)
        # 最旧的压缩部分
        compress_count = int(total_messages * self.compress_ratio)
        # 保留的原始消息
        raw_messages = self.messages[compress_count:]
        # 需要压缩的消息
        messages_to_compress = self.messages[:compress_count]

        if not messages_to_compress:
            return

        # 使用 LLM 进行压缩
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser

        compress_prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个对话摘要专家。请将以下对话历史压缩成一个简洁的摘要。

要求：
1. 保留对话的核心内容和上下文
2. 摘要长度不超过 200 字
3. 使用中文表述
4. 只输出摘要内容，不要其他解释

对话历史：
{conversation}"""),
        ])

        # 将消息列表转换为文本
        conversation_text = "\n".join([
            f"{msg.type}: {msg.content}" for msg in messages_to_compress
        ])

        compress_chain = compress_prompt | self.llm_for_compression | StrOutputParser()

        try:
            summary = compress_chain.invoke({"conversation": conversation_text})

            # 创建摘要消息（使用 SystemMessage 类型）
            from langchain_core.messages import SystemMessage
            summary_msg = SystemMessage(
                content=f"[历史对话摘要] {summary}"
            )

            # 用摘要替换旧消息
            self.messages = [summary_msg] + raw_messages

            print(f"\n📝 已压缩 {len(messages_to_compress)} 条消息为摘要")

        except Exception as e:
            print(f"⚠️ 消息压缩失败: {e}，保留原始消息")
            # 如果压缩失败，保留所有消息但只保留最近的一部分
            self.messages = self.messages[-self.buffer_size * 2:]

    def add_message(self, message: BaseMessage) -> None:
        """添加单条消息并触发压缩检查"""
        self.messages.append(message)

        # 检查是否需要压缩
        if len(self.messages) >= self.compress_threshold:
            self._compress_old_messages()

        # 持久化到数据库
        self._save_messages()

    def add_messages(self, messages: List[BaseMessage]) -> None:
        """批量添加消息"""
        self.messages.extend(messages)

        # 检查是否需要压缩
        if len(self.messages) >= self.compress_threshold:
            self._compress_old_messages()

        # 持久化到数据库
        self._save_messages()

    def clear(self) -> None:
        """清空会话历史"""
        self.messages.clear()
        Session = sessionmaker(bind=self.engine)
        session = Session()

        try:
            session.query(CompressedMessage).filter(
                CompressedMessage.session_id == self.session_id
            ).delete()
            session.commit()
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()


def get_compressed_session_history(
    session_id: str,
    engine,
    llm_for_compression,
    buffer_size: int = 10,
    compress_threshold: int = 15,
    compress_ratio: float = 0.5,
) -> CompressedChatHistory:
    """获取压缩式会话历史实例

    Args:
        session_id: 会话ID
        engine: SQLAlchemy 数据库引擎
        llm_for_compression: 用于压缩的 LLM
        buffer_size: 保留的原始对话数量
        compress_threshold: 触发压缩的总消息数阈值
        compress_ratio: 压缩比例
    """
    return CompressedChatHistory(
        session_id=session_id,
        engine=engine,
        llm_for_compression=llm_for_compression,
        buffer_size=buffer_size,
        compress_threshold=compress_threshold,
        compress_ratio=compress_ratio,
    )
