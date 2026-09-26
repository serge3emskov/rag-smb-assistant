"""Сопоставление сущностей (тарифы, продукты, услуги).

Типичная ошибка RAG в банковской документации: вопрос про тариф «Первый шаг»,
а ретривер приносит чанк про «Деловой ритм», потому что тексты почти одинаковы.
Каталог сущностей с алиасами решает две задачи:
1) распознаёт сущность в вопросе (точное совпадение алиаса или fuzzy на опечатки);
2) бустит чанки, где эта сущность упомянута, и штрафует чанки про «соседей» того же типа.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from rapidfuzz import fuzz

from .chunking import Chunk
from .retrievers import Hit
from .text import normalize as _normalize
from .text import tokenize

_PUNCT_RE = re.compile(r"[«»\"'“”„()\[\]:;,.!?]")


def normalize(text: str) -> str:
    """Для сопоставления сущностей убираем кавычки и пунктуацию: «Экспресс» == экспресс."""
    return re.sub(r"\s+", " ", _PUNCT_RE.sub(" ", _normalize(text))).strip()


@dataclass
class Entity:
    entity_id: str
    type: str
    name: str
    aliases: list[str] = field(default_factory=list)

    @property
    def surface_forms(self) -> list[str]:
        return sorted({normalize(x) for x in [self.name, *self.aliases]}, key=len, reverse=True)


@dataclass
class EntityMatch:
    entity: Entity
    surface: str
    score: float      # 100 — точное совпадение


class EntityCatalog:
    def __init__(self, entities: list[Entity], fuzzy_threshold: int = 88):
        self.entities = entities
        self.fuzzy_threshold = fuzzy_threshold

    @classmethod
    def from_yaml(cls, path: str | Path, **kw) -> "EntityCatalog":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        ents = [Entity(e["id"], e.get("type", "product"), e["name"], e.get("aliases", []))
                for e in raw.get("entities", [])]
        return cls(ents, **kw)

    # ---------------------------------------------------------- matching
    # Сравнение идёт по основам слов: «на Деловом ритме» == «Деловой ритм».
    def _forms(self, ent: Entity) -> list[tuple[str, ...]]:
        cache = self.__dict__.setdefault("_forms_cache", {})
        if ent.entity_id not in cache:
            forms = {tuple(tokenize(sf, drop_stop=False)) for sf in ent.surface_forms}
            cache[ent.entity_id] = sorted((f for f in forms if f), key=len, reverse=True)
        return cache[ent.entity_id]

    def match(self, text: str) -> list[EntityMatch]:
        toks = tokenize(text, drop_stop=False)
        found: dict[str, EntityMatch] = {}
        for ent in self.entities:
            for form in self._forms(ent):
                n = len(form)
                if any(tuple(toks[i:i + n]) == form for i in range(len(toks) - n + 1)):
                    found[ent.entity_id] = EntityMatch(ent, " ".join(form), 100.0)
                    break
        # fuzzy — для опечаток; короткие однословные формы не трогаем, иначе много ложных срабатываний
        for ent in self.entities:
            if ent.entity_id in found:
                continue
            for form in self._forms(ent):
                n, target = len(form), " ".join(form)
                if len(target) < 8:
                    continue
                for i in range(len(toks) - n + 1):
                    window = " ".join(toks[i:i + n])
                    s = fuzz.ratio(window, target)
                    if s >= self.fuzzy_threshold:
                        prev = found.get(ent.entity_id)
                        if not prev or s > prev.score:
                            found[ent.entity_id] = EntityMatch(ent, window, s)
        return self._drop_nested(list(found.values()))

    @staticmethod
    def _drop_nested(ms: list[EntityMatch]) -> list[EntityMatch]:
        """«Эквайринг» внутри «Торговый эквайринг» — оставляем более длинное совпадение."""
        keep = []
        for m in ms:
            if not any(o is not m and m.surface in o.surface and len(o.surface) > len(m.surface)
                       and o.entity.type == m.entity.type for o in ms):
                keep.append(m)
        return keep

    def tag_chunks(self, chunks: list[Chunk]) -> None:
        """entities — все упомянутые сущности; primary_entities — сущности из заголовка раздела,
        т.е. то, ЧЕМУ посвящён чанк (а не что в нём мимоходом упомянуто)."""
        for c in chunks:
            c.entities = [m.entity.entity_id for m in self.match(c.indexed_text) if m.score == 100.0]
            last_heading = c.heading.split(" > ")[-1]
            c.primary_entities = [m.entity.entity_id for m in self.match(last_heading) if m.score == 100.0]

    def by_id(self, entity_id: str) -> Entity | None:
        return next((e for e in self.entities if e.entity_id == entity_id), None)


class EntityAwareRetriever:
    """Обёртка над любым ретривером: берёт пул кандидатов и переранжирует с учётом сущностей."""

    def __init__(self, base, catalog: EntityCatalog, pool: int = 30, boost: float = 1.0, penalty: float = 0.5):
        self.base = base
        self.catalog = catalog
        self.pool = pool
        self.boost = boost
        self.penalty = penalty
        self.name = f"{base.name}+entities"
        self.chunks = base.chunks
        self.last_matches: list[EntityMatch] = []

    def search(self, query: str, k: int = 5) -> list[Hit]:
        self.last_matches = self.catalog.match(query)
        cands = self.base.search(query, self.pool)
        if not self.last_matches or not cands:
            return cands[:k]
        wanted = {m.entity.entity_id for m in self.last_matches}
        wanted_types = {m.entity.type for m in self.last_matches}
        rescored = []
        for h in cands:
            base = 1.0 / h.rank                                    # ранговая шкала, не зависит от ретривера
            primary, tags = set(h.chunk.primary_entities), set(h.chunk.entities)
            if primary & wanted:
                base *= 1 + 2 * self.boost                         # раздел посвящён нужной сущности
            elif tags & wanted:
                base *= 1 + self.boost                             # сущность упомянута в тексте
            if any((e := self.catalog.by_id(t)) and e.type in wanted_types and t not in wanted
                   for t in (primary or tags)):
                base *= self.penalty                               # раздел про «соседа» того же типа
            rescored.append((base, h))
        rescored.sort(key=lambda x: -x[0])
        return [Hit(h.chunk, s, r) for r, (s, h) in enumerate(rescored[:k], 1)]


_QUOTED_RE = re.compile(r"(?:тариф|пакет|продукт|сервис|услуга|карта|кредит)\s+[«\"]([^»\"]{3,60})[»\"]", re.I)


def suggest_entities(chunks: list[Chunk], min_count: int = 1) -> list[tuple[str, int]]:
    """Черновик каталога: названия в кавычках после «тариф/пакет/продукт…».
    Результат — отправная точка, каталог всё равно проверяется руками."""
    counts: dict[str, int] = {}
    for c in chunks:
        for m in _QUOTED_RE.finditer(c.indexed_text):
            name = m.group(1).strip()
            counts[name] = counts.get(name, 0) + 1
    return sorted(((n, k) for n, k in counts.items() if k >= min_count), key=lambda x: -x[1])
