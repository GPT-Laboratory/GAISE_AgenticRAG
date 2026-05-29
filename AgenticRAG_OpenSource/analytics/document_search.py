"""Local vector retrieval over the knowledge base in data/raw/.

Uses multilingual-e5-base embeddings (Finnish + English) + a persisted Chroma
collection. The e5 model REQUIRES a "query: " prefix at query time (and the
indexer uses "passage: ") — these must match or retrieval silently degrades.

The embedding model and Chroma collection are loaded once as module-level
singletons (lazy) so they are not reloaded per request.
"""
from __future__ import annotations

from typing import Any, Optional

from app.config import settings

# e5 prefix conventions — keep in sync with pipelines/build_vector_index.py
QUERY_PREFIX = "query: "

_model = None  # type: ignore[var-annotated]
_collection = None  # type: ignore[var-annotated]


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(settings.EMBEDDING_MODEL)
    return _model


def _get_collection():
    global _collection
    if _collection is None:
        import chromadb

        client = chromadb.PersistentClient(path=settings.VECTOR_STORE_DIR)
        # Will raise if the index has never been built — surfaced as a friendly
        # message by search_documents().
        _collection = client.get_collection(settings.VECTOR_COLLECTION)
    return _collection


def _embed_query(query: str) -> list[float]:
    model = _get_model()
    vec = model.encode(QUERY_PREFIX + query, normalize_embeddings=True)
    return vec.tolist()


def search_documents(query: str, top_k: Optional[int] = None) -> dict[str, Any]:
    """Retrieve the top-k knowledge-base chunks for a query.

    Returns {query, chunks: [{text, source, page, sheet, score}], source: [...]}.
    On a missing/empty index, returns a {message: ...} dict instead of raising.
    """
    top_k = top_k or settings.RAG_TOP_K

    try:
        collection = _get_collection()
    except Exception:
        return {
            "query": query,
            "chunks": [],
            "source": [],
            "message": (
                "The document index has not been built yet. "
                "Run: python pipelines/build_vector_index.py"
            ),
        }

    try:
        res = collection.query(
            query_embeddings=[_embed_query(query)],
            n_results=top_k,
        )
    except Exception as exc:  # pragma: no cover - defensive
        return {"query": query, "chunks": [], "source": [], "message": f"Retrieval error: {exc}"}

    documents = (res.get("documents") or [[]])[0]
    metadatas = (res.get("metadatas") or [[]])[0]
    distances = (res.get("distances") or [[]])[0]

    chunks: list[dict[str, Any]] = []
    for doc, md, dist in zip(documents, metadatas, distances):
        md = md or {}
        chunks.append(
            {
                "text": doc,
                "source": md.get("source"),
                "page": md.get("page"),
                "sheet": md.get("sheet"),
                # Chroma cosine distance -> similarity in [0, 1]
                "score": round(1.0 - float(dist), 4),
            }
        )

    sources = sorted({c["source"] for c in chunks if c.get("source")})
    return {"query": query, "chunks": chunks, "source": sources}


def is_weak(result: dict[str, Any]) -> bool:
    """Corrective-RAG gate: True when retrieval is empty or low-confidence."""
    chunks = result.get("chunks") or []
    if not chunks:
        return True
    best = max((c.get("score") or 0.0) for c in chunks)
    return best < settings.RAG_SCORE_FLOOR
