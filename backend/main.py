from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.endpoints.chat import router as chat_router
from backend.api.endpoints.health import router as health_router
from backend.api.endpoints.rag import router as rag_router
from backend.db.database import engine
from backend.db.models import Base

app = FastAPI(title="Ollama Chat API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup() -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


app.include_router(chat_router)
app.include_router(rag_router)
app.include_router(health_router)
