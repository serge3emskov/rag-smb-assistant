"""Черновик golden dataset из корпуса с помощью LLM.

Для каждого выбранного чанка модель пишет вопрос словами клиента, короткий ответ,
дословную цитату-обоснование и ключевой факт. Всё, что не удаётся проверить
по тексту чанка (цитата не дословная, факта нет во фрагменте), отбрасывается.

Черновик — отправная точка, а не эталон: его нужно просмотреть руками, убрать
тривиальные вопросы и добавить сложные (multi-hop, путаница продуктов, опечатки).
Вопросы, сгенерированные по одному чанку, лексически близки к нему и завышают
метрики поиска — поэтому в запросе модель просят перефразировать.
"""
from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from pathlib import Path

from .chunking import Chunk
from .text import normalize

QUESTION_PROMPT = """Ты составляешь тестовые вопросы для ассистента банка по документации для малого бизнеса.
По ФРАГМЕНТУ придумай один вопрос, который мог бы задать клиент — предприниматель.
Требования:
- вопрос сформулирован своими словами клиента, без дословного копирования фраз из фрагмента;
- ответ на вопрос однозначно содержится во ФРАГМЕНТЕ;
- лучше вопрос про конкретное условие: сумму, процент, срок, лимит, документ, требование.
Верни ТОЛЬКО JSON:
{{"question": "...", "answer": "короткий ответ", "quote": "дословная цитата из фрагмента 5–80 символов, подтверждающая ответ", "fact": "ключевое число или термин из ответа, дословно как во фрагменте"}}
Если во фрагменте нет содержательного условия (навигация, реклама, общие слова), верни {{"skip": true}}.

ФРАГМЕНТ ({heading}):
{text}"""

# Правдоподобные для малого бизнеса темы; перед использованием проверьте, что их нет в сохранённых страницах
UNANSWERABLE_CANDIDATES = [
    "Какая ставка по ипотеке на коммерческую недвижимость для ИП?",
    "Сколько стоит страхование товаров на складе?",
    "Можно ли открыть счёт в тайских батах?",
    "Какие условия лизинга грузовика для ИП?",
    "Сколько стоит обслуживание брокерского счёта для юрлица?",
    "Какая доходность по облигациям банка для бизнеса?",
]


def _parse_json(raw: str) -> dict | None:
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def _contains(haystack: str, needle: str) -> bool:
    n = normalize(needle).strip(" .,;:«»\"")
    return bool(n) and n in normalize(haystack)


def sample_chunks(chunks: list[Chunk], n: int, min_chars: int = 150, seed: int = 42) -> list[Chunk]:
    """Равномерно по документам: иначе один длинный PDF съест весь датасет."""
    rnd = random.Random(seed)
    by_doc = defaultdict(list)
    for c in chunks:
        if len(c.text) >= min_chars and re.search(r"\d", c.text):
            by_doc[c.doc_id].append(c)
    for lst in by_doc.values():
        rnd.shuffle(lst)
    out, docs = [], sorted(by_doc)
    while len(out) < n and any(by_doc.values()):
        for d in docs:
            if by_doc[d] and len(out) < n:
                out.append(by_doc[d].pop())
    return out


def generate_golden_draft(chunks: list[Chunk], generator, n: int = 40, seed: int = 42,
                          id_prefix: str = "s", progress: bool = True) -> tuple[list[dict], dict]:
    picked = sample_chunks(chunks, n, seed=seed)
    it = picked
    if progress:
        try:
            from tqdm.auto import tqdm
            it = tqdm(picked, desc="генерация вопросов")
        except ImportError:
            pass
    rows, stats = [], defaultdict(int)
    for c in it:
        raw = generator.chat([{"role": "user", "content": QUESTION_PROMPT.format(heading=c.heading, text=c.text)}],
                             max_new_tokens=300)
        d = _parse_json(raw)
        if not d:
            stats["bad_json"] += 1
            continue
        if d.get("skip"):
            stats["skipped_by_model"] += 1
            continue
        q, quote, fact = (d.get("question") or "").strip(), (d.get("quote") or "").strip(), (d.get("fact") or "").strip()
        if not q or not _contains(c.text, quote):
            stats["quote_not_verbatim"] += 1
            continue
        if fact and not _contains(c.text, fact):
            fact = ""
        rows.append({
            "id": f"{id_prefix}{len(rows) + 1:03d}",
            "question": q,
            "category": c.doc_id,
            "difficulty": "single",
            "expected_docs": [c.doc_id],
            "evidence": [quote],
            "expected_facts": [fact or quote],
            "expected_entities": [],
        })
        stats["accepted"] += 1
    return rows, dict(stats)


def unanswerable_rows(start: int, id_prefix: str = "s") -> list[dict]:
    return [{"id": f"{id_prefix}{start + i:03d}", "question": q, "answerable": False, "category": "без ответа",
             "difficulty": "unanswerable", "expected_docs": [], "evidence": [], "expected_facts": [],
             "expected_entities": []} for i, q in enumerate(UNANSWERABLE_CANDIDATES)]


def write_jsonl(rows: list[dict], path: str | Path, header: str | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        if header:
            for line in header.splitlines():
                f.write(f"// {line}\n")
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return path
