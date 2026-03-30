import base64
from collections.abc import Iterable
from io import BytesIO

from fastapi import APIRouter, Depends, HTTPException
from ollama import AsyncClient
from ollama._types import ResponseError
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession
try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

try:
    from ..schemas import RagIngestRequest, RagQueryRequest, RagQueryResponse
    from ...db.database import USE_PGVECTOR, get_db_session
    from ...db.models import RagChunk
    from ...services.rag import (
        Candidate,
        chunk_text,
        cosine_similarity,
        is_context_length_error,
        mmr_select,
        split_chunk,
    )
except ImportError:
    from api.schemas import RagIngestRequest, RagQueryRequest, RagQueryResponse
    from db.database import USE_PGVECTOR, get_db_session
    from db.models import RagChunk
    from services.rag import (
        Candidate,
        chunk_text,
        cosine_similarity,
        is_context_length_error,
        mmr_select,
        split_chunk,
    )

router = APIRouter(prefix="/rag", tags=["rag"])
ollama = AsyncClient()
EMBEDDING_RETRY_MIN_CHUNK_LENGTH = 200


def _to_vector_literal(values: list[float]) -> str:
    return "[" + ",".join(f"{value:.12g}" for value in values) + "]"

def _normalize_model_name(name: str) -> str:
    return name.split(":", 1)[0].strip().lower()


def _pick_embedding_model(available_models: Iterable[str], requested_model: str) -> str | None:
    available = [model for model in available_models if model]
    if not available:
        return None

    requested_normalized = _normalize_model_name(requested_model)
    for model in available:
        if _normalize_model_name(model) == requested_normalized:
            return model

    preferred = ("all-minilm", "mxbai-embed-large", "nomic-embed-text")
    for candidate in preferred:
        for model in available:
            if _normalize_model_name(model) == candidate:
                return model

    for model in available:
        normalized = _normalize_model_name(model)
        if "embed" in normalized or "minilm" in normalized:
            return model

    return None


async def _resolve_embedding_model(requested_model: str) -> str:
    tags = await ollama.list()
    available_models = [model.model for model in tags.models]
    selected = _pick_embedding_model(available_models, requested_model)
    if selected:
        return selected

    if available_models:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Embedding model '{requested_model}' is not installed. "
                f"Available models: {', '.join(available_models)}"
            ),
        )

    raise HTTPException(
        status_code=404,
        detail=(
            f"Embedding model '{requested_model}' is not installed and no local models were found. "
            "Install an embedding model (for example: all-minilm)."
        ),
    )


def _extract_pdf_text(content_base64: str) -> str:
    if PdfReader is None:
        raise HTTPException(
            status_code=500,
            detail="PDF upload requires pypdf. Install backend dependencies and restart the API.",
        )

    try:
        pdf_bytes = base64.b64decode(content_base64)
        reader = PdfReader(BytesIO(pdf_bytes))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Unable to read PDF: {exc}") from exc

    pages: list[str] = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            pages.append("")

    text = "\n\n".join(page.strip() for page in pages if page.strip()).strip()
    if not text:
        raise HTTPException(
            status_code=400,
            detail="The PDF did not contain extractable text. Try a text-based PDF or OCR it first.",
        )
    return text


@router.post("/ingest")
async def ingest_document(req: RagIngestRequest, db: AsyncSession = Depends(get_db_session)):
    try:
        embedding_model = await _resolve_embedding_model(req.embedding_model)
        content = _extract_pdf_text(req.content_base64) if req.content_base64 else req.content
        chunks = chunk_text(content)
        if not chunks:
            raise HTTPException(status_code=400, detail="Uploaded document is empty.")

        chunk_queue = list(chunks)
        stored_count = 0
        while chunk_queue:
            chunk = chunk_queue.pop(0)
            try:
                emb = await ollama.embeddings(model=embedding_model, prompt=chunk)
            except ResponseError as exc:
                if not is_context_length_error(str(exc)):
                    raise

                next_max_length = max(len(chunk) // 2, EMBEDDING_RETRY_MIN_CHUNK_LENGTH)
                smaller_chunks = split_chunk(chunk, max_length=next_max_length)
                if len(smaller_chunks) <= 1:
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            "The uploaded document contains sections that exceed the embedding "
                            "model context limit even after retrying with smaller chunks."
                        ),
                    ) from exc

                chunk_queue = smaller_chunks + chunk_queue
                continue

            row = RagChunk(
                source=req.source,
                content=chunk,
                embedding_model=embedding_model,
                embedding=emb.embedding,
                meta={**req.metadata, "chunk_index": stored_count},
            )
            db.add(row)
            stored_count += 1

        await db.commit()
        return {"status": "indexed", "source": req.source, "chunks": stored_count}
    except HTTPException:
        await db.rollback()
        raise
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
        embedding_model = await _resolve_embedding_model(req.embedding_model)
        emb = await ollama.embeddings(model=embedding_model, prompt=req.question)
        query_embedding = emb.embedding
        candidate_limit = max(req.top_k * 3, 8)
        candidates: list[Candidate]

        if USE_PGVECTOR:
            query_vector = _to_vector_literal(query_embedding)
            result = await db.execute(
                text(
                    """
                    SELECT
                        id,
                        source,
                        content,
                        embedding,
                        1 - (embedding <=> CAST(:query_embedding AS vector)) AS score
                    FROM rag_chunks
                    WHERE embedding_model = :embedding_model
                    ORDER BY embedding <=> CAST(:query_embedding AS vector)
                    LIMIT :candidate_limit
                    """
                ),
                {
                    "query_embedding": query_vector,
                    "embedding_model": embedding_model,
                    "candidate_limit": candidate_limit,
                },
            )
            rows = result.mappings().all()
            if not rows:
                raise HTTPException(
                    status_code=404,
                    detail=f"No indexed documents available for embedding model '{embedding_model}'",
                )

            candidates = [
                Candidate(
                    id=str(row["id"]),
                    source=row["source"],
                    content=row["content"],
                    score=float(row["score"]),
                    embedding=list(row["embedding"]),
                )
                for row in rows
                if float(row["score"]) >= req.min_score
            ]
        else:
            result = await db.execute(
                select(RagChunk).where(RagChunk.embedding_model == embedding_model)
            )
            rows = result.scalars().all()
            if not rows:
                raise HTTPException(
                    status_code=404,
                    detail=f"No indexed documents available for embedding model '{embedding_model}'",
                )

            candidates = [
                Candidate(
                    id=str(row.id),
                    source=row.source,
                    content=row.content,
                    score=score,
                    embedding=list(row.embedding),
                )
                for row in rows
                for score in [cosine_similarity(query_embedding, row.embedding)]
                if score >= req.min_score
            ]
            candidates = sorted(candidates, key=lambda item: item.score, reverse=True)[:candidate_limit]

        if not candidates:
            return RagQueryResponse(
                response=(
                    "I could not find a matching chunk in the uploaded documents for this question. "
                    "Try rephrasing the question or upload a more relevant document."
                ),
                model=req.model,
                contexts=[],
            )

        top_by_similarity = sorted(candidates, key=lambda c: c.score, reverse=True)
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
