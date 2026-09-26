"""Проверка обоснованности ответа (детектор галлюцинаций).

Два уровня:
* rule-based — работает всегда и мгновенно: невалидные ссылки, числа, которых нет
  в процитированных фрагментах, предложения без лексической опоры на источник;
* LLM-судья — разбивает ответ на утверждения и проверяет каждое по контексту.

Числа проверяются отдельно, потому что в банковских ответах самая дорогая
галлюцинация — неверная комиссия, лимит или ставка.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from .prompts import JUDGE_PROMPT, REFUSAL, build_context
from .text import normalize, numbers, split_sentences, tokenize

_CITE_RE = re.compile(r"\[(\d+)\]")


def is_refusal(answer: str) -> bool:
    a = normalize(answer)
    return normalize(REFUSAL).rstrip(".") in a or a.startswith("в документах нет")


def citations(answer: str) -> list[int]:
    return [int(x) for x in _CITE_RE.findall(answer)]


@dataclass
class Grounding:
    refused: bool
    cited: list[int] = field(default_factory=list)
    invalid_citations: list[int] = field(default_factory=list)
    unsupported_numbers: list[str] = field(default_factory=list)
    unsupported_sentences: list[str] = field(default_factory=list)
    judge_claims: list[dict] | None = None

    @property
    def hallucinated(self) -> bool:
        if self.refused:
            return False
        judge_bad = bool(self.judge_claims) and any(c.get("verdict") != "supported" for c in self.judge_claims)
        return bool(self.invalid_citations or self.unsupported_numbers or self.unsupported_sentences or judge_bad)


def check_grounding(answer: str, hits, min_support: float = 0.5) -> Grounding:
    if is_refusal(answer):
        return Grounding(refused=True)
    cited = citations(answer)
    invalid = sorted({c for c in cited if c < 1 or c > len(hits)})
    valid = sorted({c for c in cited if 1 <= c <= len(hits)})
    # если ссылок нет — сверяем со всем контекстом (строже нельзя: модель могла забыть скобки)
    pool = [hits[c - 1].chunk for c in valid] or [h.chunk for h in hits]
    pool_text = "\n".join(c.indexed_text for c in pool)
    pool_nums = numbers(pool_text)
    pool_toks = set(tokenize(pool_text))

    clean = _CITE_RE.sub("", answer)
    bad_nums = sorted(n for n in numbers(clean) if n not in pool_nums)

    bad_sents = []
    for s in split_sentences(clean):
        toks = [t for t in tokenize(s) if len(t) > 2]
        if len(toks) >= 3 and sum(t in pool_toks for t in toks) / len(toks) < min_support:
            bad_sents.append(s)
    return Grounding(False, cited, invalid, bad_nums, bad_sents)


def llm_judge(generator, answer: str, hits) -> list[dict] | None:
    """Утверждения ответа с вердиктами. None, если судья не вернул валидный JSON."""
    if is_refusal(answer):
        return []
    prompt = JUDGE_PROMPT.replace("{context}", build_context(hits)).replace("{answer}", _CITE_RE.sub("", answer))
    raw = generator.chat([{"role": "user", "content": prompt}], max_new_tokens=600)
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        return None
    try:
        claims = json.loads(m.group(0)).get("claims", [])
        return [c for c in claims if isinstance(c, dict) and "verdict" in c]
    except json.JSONDecodeError:
        return None
