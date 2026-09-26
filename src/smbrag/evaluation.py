"""Массовый прогон golden dataset, метрики и локализация ошибок.

Ретривер:  Hit@1, Hit@k, Recall@k (покрытие evidence), MRR, точность распознавания сущностей.
Ответ:     Fact recall, доля полностью верных ответов, hallucination rate (по отвечаемым),
           корректные отказы на неотвечаемых, ложные отказы.
Ошибки классифицируются по первому сломавшемуся этапу пайплайна — так видно,
что чинить: каталог сущностей, ретривер, промпт или модель.
"""
from __future__ import annotations

import json
import statistics
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .golden import GoldItem
from .grounding import is_refusal
from .text import normalize

ERROR_ORDER = [
    "ok",
    "entity_miss",           # сущность из вопроса не распознана
    "entity_confusion",      # распознана не та сущность
    "retrieval_miss",        # в top-k нет ни одного релевантного чанка
    "retrieval_partial",     # multi-hop: найдена только часть опорных фактов
    "false_refusal",         # контекст был, а модель отказалась
    "unsupported_claim",     # в ответе есть утверждения/числа без опоры в контексте
    "incomplete_answer",     # ответ обоснован, но не содержит всех нужных фактов
    "answered_unanswerable", # на вопрос без ответа в документах модель ответила
    "correct_refusal",
]

ERROR_HINTS = {
    "entity_miss": "добавить алиасы/опечатки в каталог сущностей",
    "entity_confusion": "развести похожие алиасы, снизить fuzzy_threshold или усилить penalty",
    "retrieval_miss": "проверить чанкинг и evidence; включить dense/reranker; расширить запрос",
    "retrieval_partial": "увеличить k или добавить декомпозицию multi-hop вопроса",
    "false_refusal": "смягчить правило отказа в промпте; проверить, что факт не разрезан между чанками",
    "unsupported_claim": "ужесточить промпт (цифры только из контекста), снизить температуру, сменить модель",
    "incomplete_answer": "разрешить более длинный ответ; проверить формулировку expected_facts",
    "answered_unanswerable": "порог отказа по score ретривера; явное правило отказа в промпте",
}


def fact_present(fact: str, answer: str) -> bool:
    a = normalize(answer).replace(" ", "")
    return any(normalize(v).replace(" ", "") in a for v in fact.split("|") if v.strip())


@dataclass
class RetrievalResult:
    id: str
    first_relevant_rank: int | None
    recall: float
    hit_at: dict[int, bool]
    entities_found: list[str] = field(default_factory=list)


def evaluate_retrieval(retriever, items: list[GoldItem], k: int = 5, ks=(1, 3, 5)) -> tuple[dict, list[RetrievalResult]]:
    rows = []
    for it in items:
        if not it.answerable:
            continue
        hits = retriever.search(it.question, max(max(ks), k))
        chunks = [h.chunk for h in hits]
        rank = next((h.rank for h in hits if it.relevant(h.chunk)), None)
        rows.append(RetrievalResult(
            it.id, rank, it.evidence_covered(chunks[:k]),
            {kk: rank is not None and rank <= kk for kk in ks},
            [m.entity.entity_id for m in getattr(retriever, "last_matches", [])],
        ))
    n = len(rows) or 1
    summary = {f"hit@{kk}": sum(r.hit_at[kk] for r in rows) / n for kk in ks}
    summary[f"recall@{k}"] = sum(r.recall for r in rows) / n
    summary["mrr"] = sum(1 / r.first_relevant_rank if r.first_relevant_rank else 0 for r in rows) / n
    with_ents = [(it, r) for it, r in zip([i for i in items if i.answerable], rows) if it.expected_entities]
    if with_ents and hasattr(retriever, "last_matches"):
        summary["entity_acc"] = sum(set(it.expected_entities) <= set(r.entities_found) for it, r in with_ents) / len(with_ents)
    return summary, rows


@dataclass
class QAResult:
    id: str
    question: str
    category: str
    difficulty: str
    answerable: bool
    answer: str
    refused: bool
    hallucinated: bool
    fact_recall: float
    first_relevant_rank: int | None
    evidence_recall: float
    entities_expected: list[str]
    entities_found: list[str]
    error: str
    retrieved: list[str]
    unsupported_numbers: list[str]
    unsupported_sentences: list[str]
    judge_claims: list | None
    latency_s: float

    @property
    def correct(self) -> bool:
        return self.error in ("ok", "correct_refusal")


def classify(it: GoldItem, refused: bool, hallucinated: bool, fact_recall: float,
             rank: int | None, ev_recall: float, found: list[str], entity_aware: bool) -> str:
    if not it.answerable:
        return "correct_refusal" if refused else "answered_unanswerable"
    if entity_aware and it.expected_entities:
        if not set(it.expected_entities) & set(found):
            return "entity_confusion" if found else "entity_miss"
    if rank is None:
        return "retrieval_miss"
    if refused:
        return "false_refusal"
    if hallucinated:
        return "unsupported_claim"
    if fact_recall < 1.0:
        return "retrieval_partial" if ev_recall < 1.0 else "incomplete_answer"
    return "ok"


def evaluate_end_to_end(assistant, items: list[GoldItem], progress: bool = True) -> list[QAResult]:
    entity_aware = hasattr(assistant.retriever, "last_matches")
    out = []
    iterator = items
    if progress:
        try:
            from tqdm.auto import tqdm
            iterator = tqdm(items, desc="golden")
        except ImportError:
            pass
    for it in iterator:
        a = assistant.ask(it.question)
        chunks = [h.chunk for h in a.hits]
        rank = next((h.rank for h in a.hits if it.relevant(h.chunk)), None)
        ev = it.evidence_covered(chunks) if it.answerable else 0.0
        refused = is_refusal(a.answer)
        fr = (sum(fact_present(f, a.answer) for f in it.expected_facts) / len(it.expected_facts)
              if it.expected_facts else float(not refused))
        g = a.grounding
        found = [m.entity.entity_id for m in a.entities]
        out.append(QAResult(
            it.id, it.question, it.category, it.difficulty, it.answerable, a.answer, refused,
            bool(g and g.hallucinated), fr, rank, ev, it.expected_entities, found,
            classify(it, refused, bool(g and g.hallucinated), fr, rank, ev, found, entity_aware),
            [h.chunk.chunk_id for h in a.hits],
            g.unsupported_numbers if g else [], g.unsupported_sentences if g else [],
            g.judge_claims if g else None, round(a.latency_s, 3),
        ))
    return out


def summarize(results: list[QAResult]) -> dict:
    ans = [r for r in results if r.answerable]
    una = [r for r in results if not r.answerable]
    answered = [r for r in ans if not r.refused]
    s = {
        "n_questions": len(results),
        "n_answerable": len(ans),
        "n_unanswerable": len(una),
        "accuracy": _mean([r.correct for r in results]),
        "answerable_accuracy": _mean([r.correct for r in ans]),
        "fact_recall": _mean([r.fact_recall for r in ans]),
        "hit@k": _mean([r.first_relevant_rank is not None for r in ans]),
        "mrr": _mean([1 / r.first_relevant_rank if r.first_relevant_rank else 0 for r in ans]),
        "evidence_recall": _mean([r.evidence_recall for r in ans]),
        "hallucination_rate": _mean([r.hallucinated for r in answered]),
        "false_refusal_rate": _mean([r.refused for r in ans]),
        "refusal_accuracy_unanswerable": _mean([r.refused for r in una]),
        "latency_p50_s": statistics.median([r.latency_s for r in results]) if results else 0,
    }
    s["errors"] = dict(sorted(Counter(r.error for r in results).items(), key=lambda x: ERROR_ORDER.index(x[0])))
    return s


def breakdown(results: list[QAResult], key: str) -> dict[str, dict]:
    groups = defaultdict(list)
    for r in results:
        groups[getattr(r, key)].append(r)
    return {g: {"n": len(rs), "accuracy": _mean([r.correct for r in rs]),
                "fact_recall": _mean([r.fact_recall for r in rs if r.answerable]),
                "hallucination_rate": _mean([r.hallucinated for r in rs if not r.refused])}
            for g, rs in sorted(groups.items())}


def _mean(xs) -> float:
    xs = list(xs)
    return round(sum(xs) / len(xs), 4) if xs else 0.0


# ------------------------------------------------------------------ отчёт
def _pct(x) -> str:
    if isinstance(x, float) and x != x:        # NaN из pandas
        return "—"
    return f"{x * 100:.1f}%" if isinstance(x, float) else str(x)


def write_report(results: list[QAResult], out_dir: str | Path, title: str = "Оценка RAG-ассистента",
                 meta: dict | None = None, ablation: list[dict] | None = None) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "results.jsonl").open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
    s = summarize(results)
    (out / "summary.json").write_text(json.dumps({"meta": meta or {}, "summary": s, "ablation": ablation},
                                                 ensure_ascii=False, indent=2), encoding="utf-8")

    L = [f"# {title}", ""]
    if meta:
        L += [" · ".join(f"**{k}**: {v}" for k, v in meta.items()), ""]
    L += ["## Итог", "", "| Метрика | Значение |", "|---|---|"]
    names = {
        "accuracy": "Доля верных (все вопросы)", "answerable_accuracy": "Доля верных (отвечаемые)",
        "fact_recall": "Fact recall", "hit@k": "Hit@k ретривера", "mrr": "MRR",
        "evidence_recall": "Recall опорных фактов", "hallucination_rate": "Hallucination rate (среди ответов)",
        "false_refusal_rate": "Ложные отказы", "refusal_accuracy_unanswerable": "Верные отказы на неотвечаемых",
        "latency_p50_s": "Латентность p50, с",
    }
    L += [f"| {names[k]} | {_pct(s[k]) if k != 'latency_p50_s' else round(s[k], 2)} |" for k in names]
    L += ["", f"Вопросов: {s['n_questions']} (отвечаемых {s['n_answerable']}, без ответа в документах {s['n_unanswerable']})", ""]

    if ablation:
        cols = list(dict.fromkeys(c for row in ablation for c in row if c != "config"))
        L += ["## Сравнение конфигураций ретривера", "", "| Конфигурация | " + " | ".join(cols) + " |",
              "|---|" + "---|" * len(cols)]
        L += [f"| {row['config']} | " + " | ".join(_pct(row[c]) if c in row else "—" for c in cols) + " |"
              for row in ablation]
        L.append("")

    L += ["## Где ломается пайплайн", "", "| Тип ошибки | Кол-во | Что чинить |", "|---|---|---|"]
    L += [f"| {e} | {n} | {ERROR_HINTS.get(e, '—')} |" for e, n in s["errors"].items()]
    for key, label in (("difficulty", "По типу вопроса"), ("category", "По категории")):
        L += ["", f"## {label}", "", "| Группа | N | Верно | Fact recall | Галлюцинации |", "|---|---|---|---|---|"]
        L += [f"| {g} | {v['n']} | {_pct(v['accuracy'])} | {_pct(v['fact_recall'])} | {_pct(v['hallucination_rate'])} |"
              for g, v in breakdown(results, key).items()]

    bad = [r for r in results if not r.correct]
    if bad:
        L += ["", "## Разбор ошибок", ""]
        for r in bad[:25]:
            L += [f"**{r.id}** · `{r.error}` · {r.question}", "",
                  f"- Ответ: {r.answer[:400]}",
                  f"- Первый релевантный чанк: {r.first_relevant_rank or 'нет в top-k'}; "
                  f"найдено {r.retrieved[:3]}"]
            if r.entities_expected:
                L.append(f"- Сущности: ожидались {r.entities_expected}, распознаны {r.entities_found}")
            if r.unsupported_numbers:
                L.append(f"- Числа без опоры в источнике: {r.unsupported_numbers}")
            L.append("")
    path = out / "report.md"
    path.write_text("\n".join(L), encoding="utf-8")
    return path
