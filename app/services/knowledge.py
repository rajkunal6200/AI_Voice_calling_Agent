"""Retrieval over company policy documents.

The ERP answers "is it in stock"; this answers "what's your returns policy" —
questions whose answers live in prose rather than in a table.

Two choices are driven by this being a *voice* agent:

*Embeddings run locally.* fastembed is a small ONNX model on the CPU, so
embedding the caller's question costs a few milliseconds and no network round
trip. A hosted embedding API would add a hop to the latency budget before the
LLM has even started, and the budget is already tight.

*Passages are short and few.* Retrieved text is read aloud. Returning three
paragraph-sized passages and asking the model to summarise gives a one-sentence
answer; returning ten long chunks gives a monologue the caller cannot follow.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Any

from app.db import execute, fetch_all
from app.errors import IntegrationError
from app.observability.logging import get_logger

if TYPE_CHECKING:
    from fastembed import TextEmbedding

log = get_logger(__name__)

# BAAI/bge-small-en-v1.5 — fastembed's default. 384 dimensions, CPU-friendly,
# and good enough for a policy corpus of a few dozen paragraphs. Changing the
# model means changing EMBED_DIM and re-indexing every document.
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
EMBED_DIM = 384

DEFAULT_TOP_K = 3
#: Cosine distance above which a match is treated as irrelevant. Without this,
#: the nearest neighbour is always returned however unrelated it is, and the
#: agent confidently reads out the wrong policy.
#:
#: Calibrated against this corpus: genuine questions ("can I send something
#: back?", "how long is the guarantee?") land at 0.26–0.39, while an unrelated
#: question ("what is the capital of France?") matched its nearest document at
#: 0.60. 0.5 sits in that gap. Re-check it when the corpus changes materially —
#: `tests/test_knowledge.py` pins both ends of the band.
MAX_DISTANCE = 0.5


@dataclass(frozen=True)
class Passage:
    title: str
    body: str
    distance: float


@lru_cache(maxsize=1)
def _model() -> TextEmbedding:
    """Load the embedding model once per process.

    The first call downloads ~130MB and takes a few seconds, which is why the
    worker warms it at startup rather than paying that cost mid-call.
    """
    from fastembed import TextEmbedding

    log.info("loading_embedding_model", model=EMBED_MODEL)
    return TextEmbedding(model_name=EMBED_MODEL)


def _embed_sync(texts: list[str]) -> list[list[float]]:
    return [vector.tolist() for vector in _model().embed(texts)]


async def embed(texts: list[str]) -> list[list[float]]:
    """Embed off the event loop — fastembed is synchronous and CPU-bound."""
    try:
        return await asyncio.to_thread(_embed_sync, texts)
    except Exception as exc:  # noqa: BLE001 — surfaced as a domain error
        raise IntegrationError(
            f"embedding failed: {exc}",
            caller_message=(
                "I can't look that up right now, but I can put you through to "
                "someone who can help."
            ),
        ) from exc


def load_model() -> None:
    """Load the embedding model synchronously.

    For the worker's prewarm hook, which runs before the process takes work
    and is not a coroutine.
    """
    _model()


async def warm_up() -> None:
    """Load the model ahead of the first call. Never raises."""
    try:
        await embed(["warm up"])
        log.info("embedding_model_ready")
    except Exception as exc:  # noqa: BLE001
        log.warning("embedding_warmup_failed", error=str(exc))


async def index_document(*, title: str, body: str) -> None:
    """Add or replace one passage in the knowledge base."""
    vector = (await embed([body]))[0]
    await execute(
        """
        INSERT INTO documents (title, body, embedding)
        VALUES (:title, :body, CAST(:embedding AS vector))
        ON CONFLICT (title) DO UPDATE
        SET body = EXCLUDED.body, embedding = EXCLUDED.embedding
        """,
        title=title,
        body=body,
        embedding=str(vector),
    )
    log.info("document_indexed", title=title)


async def search(question: str, *, top_k: int = DEFAULT_TOP_K) -> list[Passage]:
    """Passages most similar to the question, closest first.

    An empty list means "nothing relevant", which the agent should report as
    not knowing rather than filling in from its own assumptions.
    """
    vector = (await embed([question]))[0]
    rows: list[dict[str, Any]] = await fetch_all(
        """
        SELECT title, body, embedding <=> CAST(:embedding AS vector) AS distance
        FROM documents
        WHERE embedding IS NOT NULL
        ORDER BY distance
        LIMIT :top_k
        """,
        embedding=str(vector),
        top_k=top_k,
    )
    passages = [
        Passage(title=row["title"], body=row["body"], distance=float(row["distance"]))
        for row in rows
        if float(row["distance"]) <= MAX_DISTANCE
    ]
    log.info(
        "knowledge_search",
        question=question[:80],
        candidates=len(rows),
        kept=len(passages),
        best=round(passages[0].distance, 3) if passages else None,
    )
    return passages
