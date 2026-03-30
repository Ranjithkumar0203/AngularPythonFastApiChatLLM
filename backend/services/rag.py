from __future__ import annotations

from dataclasses import dataclass
from math import sqrt


@dataclass
class Candidate:
    id: str
    source: str
    content: str
    score: float
    embedding: list[float]


def chunk_text(text: str, *, chunk_size: int = 1200, overlap: int = 150) -> list[str]:
    normalized = text.replace("\r\n", "\n").strip()
    if not normalized:
        return []

    chunks: list[str] = []
    start = 0
    text_length = len(normalized)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        if end < text_length:
            split_at = normalized.rfind("\n\n", start, end)
            if split_at <= start:
                split_at = normalized.rfind("\n", start, end)
            if split_at <= start:
                split_at = normalized.rfind(" ", start, end)
            if split_at > start:
                end = split_at

        chunk = normalized[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        start = max(end - overlap, start + 1)

    return chunks


def split_chunk(chunk: str, *, max_length: int, overlap: int = 100) -> list[str]:
    normalized = chunk.strip()
    if not normalized:
        return []

    if len(normalized) <= max_length:
        return [normalized]

    if max_length <= overlap:
        overlap = max(0, max_length // 5)

    pieces: list[str] = []
    start = 0
    text_length = len(normalized)

    while start < text_length:
        end = min(start + max_length, text_length)
        if end < text_length:
            split_at = normalized.rfind("\n\n", start, end)
            if split_at <= start:
                split_at = normalized.rfind("\n", start, end)
            if split_at <= start:
                split_at = normalized.rfind(" ", start, end)
            if split_at > start:
                end = split_at

        piece = normalized[start:end].strip()
        if piece:
            pieces.append(piece)

        if end >= text_length:
            break

        start = max(end - overlap, start + 1)

    return pieces


def is_context_length_error(message: str) -> bool:
    normalized = message.lower()
    return "context length" in normalized or "input length exceeds" in normalized


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0

    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sqrt(sum(x * x for x in a))
    norm_b = sqrt(sum(y * y for y in b))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot / (norm_a * norm_b)


def mmr_select(
    query_embedding: list[float],
    candidates: list[Candidate],
    k: int = 4,
    lambda_param: float = 0.7,
) -> list[Candidate]:
    if not candidates:
        return []

    selected: list[Candidate] = []
    remaining = candidates.copy()

    while remaining and len(selected) < k:
        best = None
        best_score = float("-inf")

        for candidate in remaining:
            relevance = cosine_similarity(query_embedding, candidate.embedding)
            diversity_penalty = 0.0

            if selected:
                diversity_penalty = max(
                    cosine_similarity(candidate.embedding, picked.embedding) for picked in selected
                )

            mmr_score = (lambda_param * relevance) - ((1 - lambda_param) * diversity_penalty)
            if mmr_score > best_score:
                best_score = mmr_score
                best = candidate

        if best is None:
            break

        selected.append(best)
        remaining = [item for item in remaining if item.id != best.id]

    return selected
