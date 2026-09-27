"""Ретриверы: BM25 (лексика), TF-IDF по символьным n-граммам (офлайн-замена
эмбеддингам), dense на sentence-transformers, гибрид через Reciprocal Rank Fusion
и опциональный cross-encoder реранкер."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .chunking import Chunk
from .text import normalize, tokenize


@dataclass
class Hit:
    chunk: Chunk
    score: float
    rank: int


class Retriever(Protocol):
    name: str

    def search(self, query: str, k: int = 5) -> list[Hit]: ...


# Модели и эмбеддинги кэшируются на уровне процесса: при сравнении конфигураций
# ретривер пересоздаётся много раз, а на GPU рядом уже лежит LLM.
_MODELS: dict[tuple, object] = {}
_EMB: dict[tuple, np.ndarray] = {}


def _load(kind: str, name: str, device: str | None):
    key = (kind, name, device)
    if key not in _MODELS:
        import sentence_transformers as st
        _MODELS[key] = (st.SentenceTransformer(name, device=device) if kind == "bi"
                        else st.CrossEncoder(name, device=device, max_length=512))
    return _MODELS[key]


def clear_model_cache() -> None:
    _MODELS.clear()
    _EMB.clear()


def _top(chunks: list[Chunk], scores: np.ndarray, k: int) -> list[Hit]:
    idx = np.argsort(-scores)[:k]
    return [Hit(chunks[i], float(scores[i]), r) for r, i in enumerate(idx, 1)]


class BM25Retriever:
    name = "bm25"

    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        from rank_bm25 import BM25Okapi

        self.chunks = chunks
        self.bm25 = BM25Okapi([tokenize(c.indexed_text) for c in chunks], k1=k1, b=b)

    def scores(self, query: str) -> np.ndarray:
        return np.asarray(self.bm25.get_scores(tokenize(query)))

    def search(self, query: str, k: int = 5) -> list[Hit]:
        return _top(self.chunks, self.scores(query), k)


class TfidfRetriever:
    """Символьные n-граммы устойчивы к морфологии и опечаткам; работает без GPU и интернета."""

    name = "tfidf"

    def __init__(self, chunks: list[Chunk]):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.chunks = chunks
        self.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=1)
        self.matrix = self.vec.fit_transform([normalize(c.indexed_text) for c in chunks])

    def scores(self, query: str) -> np.ndarray:
        q = self.vec.transform([normalize(query)])
        return (self.matrix @ q.T).toarray().ravel()

    def search(self, query: str, k: int = 5) -> list[Hit]:
        return _top(self.chunks, self.scores(query), k)


class DenseRetriever:
    """Эмбеддинги sentence-transformers. Для e5 нужны префиксы query:/passage:."""

    name = "dense"

    def __init__(self, chunks: list[Chunk], model_name: str = "intfloat/multilingual-e5-base",
                 batch_size: int = 32, device: str | None = None):
        self.chunks = chunks
        self.model = _load("bi", model_name, device)
        self.is_e5 = "e5" in model_name.lower()
        key = (model_name, tuple(c.chunk_id for c in chunks), hash(tuple(c.indexed_text for c in chunks)))
        if key not in _EMB:
            passages = [("passage: " if self.is_e5 else "") + c.indexed_text for c in chunks]
            _EMB[key] = self.model.encode(passages, batch_size=batch_size, normalize_embeddings=True,
                                          show_progress_bar=len(passages) > 200)
        self.emb = _EMB[key]

    def scores(self, query: str) -> np.ndarray:
        q = self.model.encode([("query: " if self.is_e5 else "") + query], normalize_embeddings=True)
        return (self.emb @ q.T).ravel()

    def search(self, query: str, k: int = 5) -> list[Hit]:
        return _top(self.chunks, self.scores(query), k)


class HybridRetriever:
    """Reciprocal Rank Fusion: score = Σ 1 / (rrf_k + rank_i). Не требует калибровки шкал."""

    def __init__(self, retrievers: list, rrf_k: int = 60, pool: int = 30):
        self.retrievers = retrievers
        self.rrf_k = rrf_k
        self.pool = pool
        self.name = "hybrid(" + "+".join(r.name for r in retrievers) + ")"
        self.chunks = retrievers[0].chunks

    def search(self, query: str, k: int = 5) -> list[Hit]:
        fused: dict[str, float] = {}
        by_id: dict[str, Chunk] = {}
        for r in self.retrievers:
            for h in r.search(query, self.pool):
                fused[h.chunk.chunk_id] = fused.get(h.chunk.chunk_id, 0.0) + 1.0 / (self.rrf_k + h.rank)
                by_id[h.chunk.chunk_id] = h.chunk
        ranked = sorted(fused.items(), key=lambda x: -x[1])[:k]
        return [Hit(by_id[cid], s, r) for r, (cid, s) in enumerate(ranked, 1)]


class Reranker:
    """Cross-encoder поверх кандидатов ретривера (например, BAAI/bge-reranker-v2-m3)."""

    def __init__(self, base, model_name: str = "BAAI/bge-reranker-v2-m3", pool: int = 20, device: str | None = None):
        self.base = base
        self.model = _load("cross", model_name, device)
        self.pool = pool
        self.name = f"{base.name}+rerank"
        self.chunks = base.chunks

    def search(self, query: str, k: int = 5) -> list[Hit]:
        cands = self.base.search(query, self.pool)
        if not cands:
            return []
        scores = self.model.predict([(query, h.chunk.indexed_text) for h in cands])
        order = np.argsort(-np.asarray(scores))[:k]
        return [Hit(cands[i].chunk, float(scores[i]), r) for r, i in enumerate(order, 1)]


def build_retriever(chunks: list[Chunk], cfg: dict):
    """cfg['retriever']: bm25 | tfidf | dense | hybrid; для hybrid — список в cfg['hybrid']."""
    kind = cfg.get("retriever", "hybrid")
    dense_cfg = cfg.get("dense", {})

    def make(name: str):
        if name == "bm25":
            return BM25Retriever(chunks)
        if name == "tfidf":
            return TfidfRetriever(chunks)
        if name == "dense":
            return DenseRetriever(chunks, dense_cfg.get("model", "intfloat/multilingual-e5-base"),
                                  device=dense_cfg.get("device"))
        raise ValueError(f"Неизвестный ретривер: {name}")

    base = (HybridRetriever([make(n) for n in cfg.get("hybrid", ["bm25", "tfidf"])], rrf_k=cfg.get("rrf_k", 60))
            if kind == "hybrid" else make(kind))
    rr = cfg.get("reranker")
    if rr and rr.get("enabled"):
        base = Reranker(base, rr.get("model", "BAAI/bge-reranker-v2-m3"), pool=rr.get("pool", 20),
                        device=dense_cfg.get("device"))
    return base
