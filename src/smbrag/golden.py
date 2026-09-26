"""Golden dataset: формат, загрузка, разметка релевантности.

Формат строки JSONL:
{
  "id": "q007",
  "question": "Сколько стоит обслуживание на тарифе Лёгкий старт?",
  "answerable": true,
  "category": "тарифы",                 # для разреза метрик
  "difficulty": "single",               # single | multi_hop | paraphrase | typo | entity_confusion | unanswerable
  "expected_docs": ["tarify_rko"],
  "evidence": ["первый шаг", "0 ₽ в месяц"],   # подстроки, по которым чанк считается релевантным
  "expected_facts": ["0 ₽|0 руб|бесплатно"],     # что обязано быть в ответе; «|» — варианты
  "expected_entities": ["tariff_first_step"]
}

Релевантность чанка: чанк из expected_docs, содержащий хотя бы одну строку evidence
(если evidence пуст — любой чанк документа). Recall@k считается по покрытию evidence:
multi-hop вопрос требует найти все опорные факты, а не один чанк.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .chunking import Chunk
from .text import normalize


@dataclass
class GoldItem:
    id: str
    question: str
    answerable: bool = True
    category: str = "general"
    difficulty: str = "single"
    expected_docs: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    expected_facts: list[str] = field(default_factory=list)
    expected_entities: list[str] = field(default_factory=list)

    def relevant(self, chunk: Chunk) -> bool:
        if self.expected_docs and chunk.doc_id not in self.expected_docs:
            return False
        if not self.evidence:
            return bool(self.expected_docs)
        text = normalize(chunk.indexed_text)
        return any(normalize(e) in text for e in self.evidence)

    def evidence_covered(self, chunks: list[Chunk]) -> float:
        """Доля evidence-строк, найденных в релевантных чанках выдачи."""
        if not self.evidence:
            return 1.0 if any(self.relevant(c) for c in chunks) else 0.0
        texts = [normalize(c.indexed_text) for c in chunks if not self.expected_docs or c.doc_id in self.expected_docs]
        return sum(any(normalize(e) in t for t in texts) for e in self.evidence) / len(self.evidence)


def load_golden(path: str | Path) -> list[GoldItem]:
    items = []
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        try:
            items.append(GoldItem(**json.loads(line)))
        except (json.JSONDecodeError, TypeError) as e:
            raise ValueError(f"{path}:{n}: {e}") from e
    ids = [i.id for i in items]
    dup = {x for x in ids if ids.count(x) > 1}
    if dup:
        raise ValueError(f"Повторяющиеся id в golden dataset: {sorted(dup)}")
    return items


def validate_golden(items: list[GoldItem], chunks: list[Chunk]) -> list[str]:
    """Проверка разметки против корпуса: несуществующие документы и evidence,
    которых нет ни в одном чанке, — частая причина «ложных» провалов ретривера."""
    problems = []
    doc_ids = {c.doc_id for c in chunks}
    for it in items:
        for d in it.expected_docs:
            if d not in doc_ids:
                problems.append(f"{it.id}: документ '{d}' отсутствует в корпусе")
        for e in it.evidence:
            if not any(normalize(e) in normalize(c.indexed_text) for c in chunks
                       if not it.expected_docs or c.doc_id in it.expected_docs):
                problems.append(f"{it.id}: evidence '{e}' не найден в документах {it.expected_docs}")
        if it.answerable and not it.expected_facts:
            problems.append(f"{it.id}: у отвечаемого вопроса нет expected_facts")
    return problems
