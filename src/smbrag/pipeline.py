"""Сборка ассистента из конфига: документы → чанки → индекс → (сущности) → генерация."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .chunking import Chunk, chunk_documents
from .entities import EntityAwareRetriever, EntityCatalog, EntityMatch
from .grounding import Grounding, check_grounding, llm_judge
from .llm import build_generator
from .loaders import load_dir
from .retrievers import Hit, build_retriever


@dataclass
class Answer:
    question: str
    answer: str
    hits: list[Hit]
    entities: list[EntityMatch] = field(default_factory=list)
    grounding: Grounding | None = None
    latency_s: float = 0.0

    def pretty(self) -> str:
        lines = [f"В: {self.question}", f"О: {self.answer}"]
        if self.entities:
            lines.append("Сущности: " + ", ".join(f"{m.entity.name} ({m.score:.0f})" for m in self.entities))
        for i, h in enumerate(self.hits, 1):
            lines.append(f"  [{i}] {h.chunk.chunk_id} | {h.chunk.heading[:90]}")
        g = self.grounding
        if g and g.hallucinated:
            lines.append(f"  ⚠ возможная галлюцинация: числа {g.unsupported_numbers}, "
                         f"ссылки {g.invalid_citations}, предложения {len(g.unsupported_sentences)}")
        return "\n".join(lines)


class Assistant:
    def __init__(self, chunks: list[Chunk], retriever, generator, k: int = 5, judge=None):
        self.chunks = chunks
        self.retriever = retriever
        self.generator = generator
        self.k = k
        self.judge = judge

    @classmethod
    def from_config(cls, cfg: dict | str | Path, generator=None, **overrides) -> "Assistant":
        if not isinstance(cfg, dict):
            cfg = yaml.safe_load(Path(cfg).read_text(encoding="utf-8"))
        cfg = {**cfg, **overrides}
        docs = load_dir(cfg["docs_dir"])
        ch = cfg.get("chunking", {})
        chunks = chunk_documents(docs, ch.get("max_chars", 900), ch.get("overlap_sents", 1))
        retr = build_retriever(chunks, cfg.get("retrieval", {}))
        ent_cfg = cfg.get("entities", {})
        if ent_cfg.get("enabled") and ent_cfg.get("catalog"):
            catalog = EntityCatalog.from_yaml(ent_cfg["catalog"], fuzzy_threshold=ent_cfg.get("fuzzy_threshold", 88))
            catalog.tag_chunks(chunks)
            retr = EntityAwareRetriever(retr, catalog, boost=ent_cfg.get("boost", 1.0),
                                        penalty=ent_cfg.get("penalty", 0.5))
        gen = generator or build_generator(cfg.get("generation", {}))
        judge = gen if cfg.get("evaluation", {}).get("llm_judge") else None
        return cls(chunks, retr, gen, cfg.get("retrieval", {}).get("k", 5), judge)

    def retrieve(self, question: str, k: int | None = None) -> list[Hit]:
        return self.retriever.search(question, k or self.k)

    def ask(self, question: str) -> Answer:
        t0 = time.perf_counter()
        hits = self.retrieve(question)
        ents = list(getattr(self.retriever, "last_matches", []))
        text = self.generator.generate(question, hits)
        g = check_grounding(text, hits)
        if self.judge is not None:
            g.judge_claims = llm_judge(self.judge, text, hits)
        return Answer(question, text, hits, ents, g, time.perf_counter() - t0)
