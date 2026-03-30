from typing import Any, Literal

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    prompt: str
    model: str = "qwen2.5:7b"
    session_id: str = Field(default="default")
    system_prompt: str = Field(default="You are a helpful AI assistant.")


class ChatResponse(BaseModel):
    response: str
    model: str


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class MultiTurnRequest(BaseModel):
    messages: list[Message]
    model: str = "qwen2.5:7b"


class RagIngestRequest(BaseModel):
    source: str
    content: str = ""
    content_base64: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    embedding_model: str = "all-minilm"


class RagQueryRequest(BaseModel):
    question: str
    model: str = "qwen2.5:7b"
    embedding_model: str = "all-minilm"
    top_k: int = 4
    min_score: float = 0.2


class RagQueryResponse(BaseModel):
    response: str
    model: str
    contexts: list[dict[str, Any]]
