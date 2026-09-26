from pathlib import Path

import pytest

from smbrag.chunking import chunk_documents
from smbrag.entities import EntityAwareRetriever, EntityCatalog
from smbrag.evaluation import classify, evaluate_end_to_end, fact_present, summarize
from smbrag.golden import GoldItem, load_golden, validate_golden
from smbrag.grounding import check_grounding, is_refusal
from smbrag.llm import ExtractiveBaseline
from smbrag.loaders import load_dir, load_html
from smbrag.pipeline import Assistant
from smbrag.prompts import REFUSAL
from smbrag.retrievers import BM25Retriever, Hit, HybridRetriever, TfidfRetriever
from smbrag.text import normalize, numbers, tokenize

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "data" / "demo"


@pytest.fixture(scope="module")
def chunks():
    return chunk_documents(load_dir(DEMO / "docs"), max_chars=700)


@pytest.fixture(scope="module")
def catalog(chunks):
    cat = EntityCatalog.from_yaml(DEMO / "entities.yaml", fuzzy_threshold=85)
    cat.tag_chunks(chunks)
    return cat


# ------------------------------------------------------------------ text
def test_normalize_numbers_and_yo():
    assert normalize("Расчёты  по счёту — 1 490 ₽") == "расчеты по счету — 1490 ₽"
    assert "1490" in numbers("плата 1 490 ₽ в месяц")
    assert numbers("1,5%") == {"1.5"}


def test_tokenize_stems_inflections():
    assert tokenize("Деловом ритме") == tokenize("Деловой ритм")


# ------------------------------------------------------------------ loaders
def test_html_loader_strips_noise_and_keeps_headings(tmp_path):
    html = """<html><head><title>Тарифы</title></head><body>
    <header>Меню сайта</header><nav>Навигация</nav>
    <main><h2>Тариф «Старт»</h2><p>Обслуживание — 0 ₽ в месяц.</p>
    <h2>Тариф «Про»</h2><ul><li>Платежи без ограничений</li></ul></main>
    <div class="cookie-banner">Мы используем cookie</div><script>var x=1</script></body></html>"""
    p = tmp_path / "t.html"
    p.write_text(html, encoding="utf-8")
    doc = load_html(p)
    text = doc.text
    assert "Тарифы > Тариф «Старт»" in text and "0 ₽ в месяц" in text
    assert "cookie" not in text and "Навигация" not in text and "var x" not in text


def test_chunks_do_not_cross_sections(chunks):
    for c in chunks:
        if "Первый шаг" in c.heading:
            assert "1 490" not in c.text          # цена «Активных расчётов» не попала в чанк «Первого шага»


# ------------------------------------------------------------------ entities
@pytest.mark.parametrize("q,expected", [
    ("Сколько стоит тариф Первый шаг?", "tariff_first_step"),
    ("Сколько платежей на Деловом ритме?", "tariff_rhythm"),       # падеж
    ("Сколько стоит тариф Делавой ритм?", "tariff_rhythm"),        # опечатки
    ("Под какой процент кредит без залога?", "credit_express"),        # алиас
    ("Ставка по СБП", "acq_sbp"),
])
def test_entity_matching(catalog, q, expected):
    assert expected in [m.entity.entity_id for m in catalog.match(q)]


def test_no_false_entity(catalog):
    assert catalog.match("Какие документы нужны для открытия счёта?") == []


def test_entity_retriever_prefers_right_tariff(chunks, catalog):
    r = EntityAwareRetriever(HybridRetriever([BM25Retriever(chunks), TfidfRetriever(chunks)]), catalog)
    top = r.search("Сколько стоит платёж юрлицу на Первом шаге после бесплатных?", 1)[0]
    assert "tariff_first_step" in top.chunk.primary_entities


# ------------------------------------------------------------------ grounding
def test_grounding_flags_invented_number(chunks):
    hit = Hit(next(c for c in chunks if "Первый шаг" in c.heading), 1.0, 1)
    ok = check_grounding("Обслуживание стоит 0 ₽ в месяц [1].", [hit])
    bad = check_grounding("Обслуживание стоит 290 ₽ в месяц [1].", [hit])
    assert not ok.hallucinated
    assert bad.hallucinated and "290" in bad.unsupported_numbers


def test_grounding_flags_invalid_citation(chunks):
    hit = Hit(chunks[0], 1.0, 1)
    assert check_grounding("Ответ [3].", [hit]).invalid_citations == [3]


def test_refusal_detection():
    assert is_refusal(REFUSAL)
    assert check_grounding(REFUSAL, []).refused


# ------------------------------------------------------------------ evaluation
def test_fact_present_alternatives():
    assert fact_present("1490", "плата — 1 490 ₽")
    assert fact_present("0 ₽|бесплатно", "Обслуживание бесплатно")
    assert not fact_present("99", "49 ₽ за платёж")


def test_classify_order():
    it = GoldItem("x", "q", expected_entities=["e1"], expected_facts=["1"])
    assert classify(it, False, False, 1.0, 1, 1.0, [], True) == "entity_miss"
    assert classify(it, False, False, 1.0, None, 0, ["e1"], True) == "retrieval_miss"
    assert classify(it, True, False, 0.0, 1, 1.0, ["e1"], True) == "false_refusal"
    assert classify(it, False, True, 1.0, 1, 1.0, ["e1"], True) == "unsupported_claim"
    una = GoldItem("y", "q", answerable=False)
    assert classify(una, False, False, 0, None, 0, [], True) == "answered_unanswerable"


def test_golden_is_consistent_with_corpus(chunks):
    assert validate_golden(load_golden(ROOT / "eval" / "golden_demo.jsonl"), chunks) == []


def test_end_to_end_baseline_quality():
    """Регрессионный порог: офлайн-baseline не должен деградировать."""
    import yaml
    cfg = yaml.safe_load((ROOT / "configs" / "demo.yaml").read_text(encoding="utf-8"))
    cfg["docs_dir"] = str(DEMO / "docs")
    cfg["entities"]["catalog"] = str(DEMO / "entities.yaml")
    asst = Assistant.from_config(cfg, generator=ExtractiveBaseline())
    s = summarize(evaluate_end_to_end(asst, load_golden(ROOT / "eval" / "golden_demo.jsonl"), progress=False))
    assert s["hit@k"] >= 0.95
    assert s["mrr"] >= 0.9
    assert s["hallucination_rate"] <= 0.05
