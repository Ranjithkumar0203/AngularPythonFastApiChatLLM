import json

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from ollama import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api.schemas import ChatRequest, ChatResponse, MultiTurnRequest
from backend.db.database import get_db_session
from backend.db.models import ChatMessage

router = APIRouter(prefix="/chat", tags=["chat"])
ollama = AsyncClient()


async def save_message(
    db: AsyncSession,
    *,
    session_id: str,
    role: str,
    content: str,
    model: str,
) -> None:
    db.add(
        ChatMessage(
            session_id=session_id,
            role=role,
            content=content,
            model=model,
        )
    )
    await db.commit()


@router.post("", response_model=ChatResponse)
async def chat(req: ChatRequest):
    try:
        result = await ollama.generate(
            model=req.model,
            prompt=req.prompt,
            system=req.system_prompt,
        )
        return ChatResponse(response=result.response, model=req.model)
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))


@router.post("/stream")
async def chat_stream(req: ChatRequest, db: AsyncSession = Depends(get_db_session)):
    async def token_generator():
        assistant_tokens: list[str] = []
        try:
            await save_message(
                db,
                session_id=req.session_id,
                role="system",
                content=req.system_prompt,
                model=req.model,
            )
            await save_message(
                db,
                session_id=req.session_id,
                role="user",
                content=req.prompt,
                model=req.model,
            )

            async for chunk in await ollama.generate(
                model=req.model,
                prompt=req.prompt,
                system=req.system_prompt,
                stream=True,
            ):
                token = chunk.response
                if token:
                    assistant_tokens.append(token)
                    yield f"data: {json.dumps({'token': token})}\n\n"

                if chunk.done:
                    break

            full_assistant_response = "".join(assistant_tokens).strip()
            if full_assistant_response:
                await save_message(
                    db,
                    session_id=req.session_id,
                    role="assistant",
                    content=full_assistant_response,
                    model=req.model,
                )
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(token_generator(), media_type="text/event-stream")


@router.post("/multi", response_model=ChatResponse)
async def chat_multi(req: MultiTurnRequest):
    try:
        result = await ollama.chat(
            model=req.model,
            messages=[message.model_dump() for message in req.messages],
        )
        return ChatResponse(response=result.message.content, model=req.model)
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))
