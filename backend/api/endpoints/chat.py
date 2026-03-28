import json

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from ollama import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

try:
    from ..schemas import ChatRequest, ChatResponse, MultiTurnRequest
    from ...db.database import AsyncSessionLocal, get_db_session
    from ...db.models import ChatMessage
except ImportError:
    from api.schemas import ChatRequest, ChatResponse, MultiTurnRequest
    from db.database import AsyncSessionLocal, get_db_session
    from db.models import ChatMessage

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


@router.websocket("/ws")
async def chat_stream_ws(websocket: WebSocket):
    await websocket.accept()

    try:
        payload = await websocket.receive_json()
        req = ChatRequest(**payload)
    except Exception as exc:
        await websocket.send_text(json.dumps({"error": f"Invalid request: {exc}"}))
        await websocket.close(code=1003)
        return

    async with AsyncSessionLocal() as db:
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
                    await websocket.send_text(json.dumps({"token": token}))

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

            await websocket.send_text(json.dumps({"done": True}))
        except WebSocketDisconnect:
            return
        except Exception as exc:
            await websocket.send_text(json.dumps({"error": str(exc)}))
        finally:
            await websocket.close()


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
