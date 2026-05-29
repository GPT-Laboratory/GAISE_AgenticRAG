from typing import Any, List, Literal, Optional
from pydantic import BaseModel


class Message(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


class ChatRequest(BaseModel):
    message: str
    history: List[Message] = []


class ToolCall(BaseModel):
    tool_name: str
    arguments: dict[str, Any]
    result: dict[str, Any]


class ChatResponse(BaseModel):
    answer: str
    tool_calls: List[ToolCall] = []
    chart_data: Optional[dict[str, Any]] = None