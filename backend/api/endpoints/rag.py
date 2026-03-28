from fastapi import APIRouter, Depends, HTTPException
from ollama import AsyncClient
from ollama._types import ResponseError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

try:
    from ..schemas import RagIngestRequest, RagQueryRequest, RagQueryResponse
    from ...db.database import get_db_session
    from ...db.models import RagChunk
    from ...services.rag import Candidate, cosine_similarity, mmr_select
except ImportError:
    from api.schemas import RagIngestRequest, RagQueryRequest, RagQueryResponse
    from db.database import get_db_session
    from db.models import RagChunk
    from services.rag import Candidate, cosine_similarity, mmr_select

router = APIRouter(prefix="/rag", tags=["rag"])
ollama = AsyncClient()


@router.post("/ingest")
async def ingest_document(req: RagIngestRequest, db: AsyncSession = Depends(get_db_session)):
    try:
        emb = await ollama.embeddings(model=req.embedding_model, prompt=req.content)
        vector = emb.embedding

        row = RagChunk(
            source=req.source,
            content=req.content,
            embedding_model=req.embedding_model,
            embedding=vector,
            meta=req.metadata,
        )
        db.add(row)
        await db.commit()
        return {"status": "indexed", "source": req.source}
    except ResponseError as e:
        await db.rollback()
        status_code = 404 if e.status_code == 404 else 502
        raise HTTPException(status_code=status_code, detail=str(e))
    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=502, detail=str(e))


@router.post("/query", response_model=RagQueryResponse)
async def rag_query(req: RagQueryRequest, db: AsyncSession = Depends(get_db_session)):
    try:
        emb = await ollama.embeddings(model=req.embedding_model, prompt=req.question)
        query_embedding = emb.embedding

        result = await db.execute(
            select(RagChunk).where(RagChunk.embedding_model == req.embedding_model)
        )
        rows = result.scalars().all()
        if not rows:
            raise HTTPException(
                status_code=404,
                detail=f"No indexed documents available for embedding model '{req.embedding_model}'",
            )

        candidates = [
            Candidate(
                id=str(row.id),
                source=row.source,
                content=row.content,
                score=cosine_similarity(query_embedding, row.embedding),
                embedding=row.embedding,
            )
            for row in rows
        ]

        top_by_similarity = sorted(candidates, key=lambda c: c.score, reverse=True)[: max(req.top_k * 3, 8)]
        selected = mmr_select(query_embedding, top_by_similarity, k=req.top_k, lambda_param=0.75)

        context_block = "\n\n".join(
            [f"[Source: {item.source}]\n{item.content}" for item in selected]
        )
        prompt = (
            "You are a grounded assistant. Use only the provided context. "
            "If context is insufficient, explicitly say what is missing.\n\n"
            f"Context:\n{context_block}\n\n"
            f"Question: {req.question}"
        )

        llm_result = await ollama.generate(model=req.model, prompt=prompt)
        return RagQueryResponse(
            response=llm_result.response,
            model=req.model,
            contexts=[{"source": item.source, "score": item.score} for item in selected],
        )
    except HTTPException:
        raise
    except ResponseError as e:
        status_code = 404 if e.status_code == 404 else 502
        raise HTTPException(status_code=status_code, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))
