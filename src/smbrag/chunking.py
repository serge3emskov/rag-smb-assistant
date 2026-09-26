"""Чанкинг с учётом структуры: чанк не пересекает границу раздела,
а к тексту добавляется путь заголовков (contextual chunk header)."""
from __future__ import annotations

from dataclasses import dataclass, field

from .loaders import Document
from .text import split_sentences


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    heading: str
    text: str
    source: str
    entities: list[str] = field(default_factory=list)          # заполняется EntityCatalog.tag_chunks
    primary_entities: list[str] = field(default_factory=list)  # сущности из заголовка раздела

    @property
    def indexed_text(self) -> str:
        """Что попадает в индекс: заголовок раздела + текст."""
        return f"{self.heading}\n{self.text}"


def chunk_documents(docs: list[Document], max_chars: int = 900, overlap_sents: int = 1) -> list[Chunk]:
    chunks: list[Chunk] = []
    for doc in docs:
        n = 0
        for sec in doc.sections:
            sents = split_sentences(sec.text) or [sec.text]
            window: list[str] = []
            for sent in sents:
                if window and len(" ".join(window)) + len(sent) > max_chars:
                    chunks.append(_mk(doc, sec.heading, window, n))
                    n += 1
                    window = window[-overlap_sents:] if overlap_sents else []
                window.append(sent)
            if window:
                chunks.append(_mk(doc, sec.heading, window, n))
                n += 1
    return chunks


def _mk(doc: Document, heading: str, sents: list[str], n: int) -> Chunk:
    return Chunk(
        chunk_id=f"{doc.doc_id}#{n}",
        doc_id=doc.doc_id,
        title=doc.title,
        heading=heading,
        text="\n".join(sents),
        source=doc.source,
    )
