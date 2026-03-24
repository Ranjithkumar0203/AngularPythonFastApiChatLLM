from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from ollama import AsyncClient          # official async client
import json

app = FastAPI(title="Ollama Chat API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# One shared async client for the whole app lifetime
ollama = AsyncClient()


# ── Schemas ────────────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    prompt: str
    model: str = "qwen2.5:7b"


class ChatResponse(BaseModel):
    response: str
    model: str


# ── Non-streaming endpoint ─────────────────────────────────────────────────────
@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """Await the full response from Ollama — no httpx needed."""
    try:
        result = await ollama.generate(
            model=req.model,
            prompt=req.prompt,
        )
        return ChatResponse(response=result.response, model=req.model)

    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))


# ── Streaming endpoint (SSE) ───────────────────────────────────────────────────
@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """Stream tokens back as Server-Sent Events using async for."""

    async def token_generator():
        try:
            # ollama.generate with stream=True returns an async iterator
            async for chunk in await ollama.generate(
                model=req.model,
                prompt=req.prompt,
                stream=True,
            ):
                token = chunk.response           # each chunk has .response
                yield f"data: {json.dumps({'token': token})}\n\n"

                if chunk.done:
                    break
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(token_generator(), media_type="text/event-stream")


# ── Chat (multi-turn) endpoint ─────────────────────────────────────────────────
class Message(BaseModel):
    role: str       # "user" | "assistant"
    content: str

class MultiTurnRequest(BaseModel):
    messages: list[Message]
    model: str = "qwen2.5:7b"

@app.post("/chat/multi")
async def chat_multi(req: MultiTurnRequest):
    """Multi-turn conversation using ollama.chat()"""
    try:
        result = await ollama.chat(
            model=req.model,
            messages=[m.model_dump() for m in req.messages],
        )
        reply = result.message.content
        return {"response": reply, "model": req.model}

    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))


# ── Health / list local models ─────────────────────────────────────────────────
@app.get("/health")
async def health():
    try:
        tags = await ollama.list()                          # returns ListResponse
        models = [m.model for m in tags.models]
        return {"status": "ok", "models": models}
    except Exception:
        return {"status": "ollama_unreachable", "models": []}
