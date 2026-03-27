from fastapi import APIRouter
from ollama import AsyncClient

router = APIRouter(tags=["health"])
ollama = AsyncClient()


@router.get("/health")
async def health():
    try:
        tags = await ollama.list()
        models = [m.model for m in tags.models]
        return {"status": "ok", "models": models}
    except Exception:
        return {"status": "ollama_unreachable", "models": []}
