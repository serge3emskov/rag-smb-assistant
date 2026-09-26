"""Командная строка.

python -m smbrag ask     -c configs/demo.yaml "Сколько стоит тариф Первый шаг?"
python -m smbrag eval    -c configs/demo.yaml --golden eval/golden_demo.jsonl --out reports/demo
python -m smbrag validate -c configs/demo.yaml --golden eval/golden_demo.jsonl
python -m smbrag suggest-entities -c configs/sber.yaml
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import yaml

from .chunking import chunk_documents
from .entities import EntityAwareRetriever, EntityCatalog, suggest_entities
from .evaluation import evaluate_end_to_end, evaluate_retrieval, summarize, write_report
from .golden import load_golden, validate_golden
from .loaders import load_dir
from .pipeline import Assistant
from .retrievers import build_retriever


def _cfg(path: str) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def _chunks(cfg):
    ch = cfg.get("chunking", {})
    return chunk_documents(load_dir(cfg["docs_dir"]), ch.get("max_chars", 900), ch.get("overlap_sents", 1))


def retrieval_ablation(cfg: dict, items, variants: list[str] | None = None, k: int = 5) -> list[dict]:
    """Сравнение ретриверов на одном golden dataset (быстро: без генерации)."""
    variants = variants or cfg.get("evaluation", {}).get("ablation", ["bm25", "tfidf", "hybrid", "hybrid+entities"])
    chunks = _chunks(cfg)
    catalog = None
    ent = cfg.get("entities", {})
    if ent.get("catalog"):
        catalog = EntityCatalog.from_yaml(ent["catalog"], fuzzy_threshold=ent.get("fuzzy_threshold", 88))
        catalog.tag_chunks(chunks)
    rows = []
    for v in variants:
        base_name, _, suffix = v.partition("+")
        rcfg = copy.deepcopy(cfg.get("retrieval", {}))
        if base_name.startswith("hybrid"):
            rcfg["retriever"] = "hybrid"
        else:
            rcfg["retriever"] = base_name
        rcfg["reranker"] = {**rcfg.get("reranker", {}), "enabled": "rerank" in v}
        r = build_retriever(chunks, rcfg)
        if "entities" in suffix and catalog:
            r = EntityAwareRetriever(r, catalog, boost=ent.get("boost", 1.0), penalty=ent.get("penalty", 0.5))
        summary, _ = evaluate_retrieval(r, items, k=k)
        rows.append({"config": v, **{kk: round(vv, 4) for kk, vv in summary.items()}})
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(prog="smbrag")
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("ask")
    a.add_argument("-c", "--config", required=True)
    a.add_argument("question", nargs="+")

    e = sub.add_parser("eval")
    e.add_argument("-c", "--config", required=True)
    e.add_argument("--golden", required=True)
    e.add_argument("--out", default="reports/latest")
    e.add_argument("--no-ablation", action="store_true")

    v = sub.add_parser("validate")
    v.add_argument("-c", "--config", required=True)
    v.add_argument("--golden", required=True)

    s = sub.add_parser("suggest-entities")
    s.add_argument("-c", "--config", required=True)

    args = p.parse_args(argv)
    cfg = _cfg(args.config)

    if args.cmd == "ask":
        print(Assistant.from_config(cfg).ask(" ".join(args.question)).pretty())
    elif args.cmd == "validate":
        problems = validate_golden(load_golden(args.golden), _chunks(cfg))
        print("\n".join(problems) or "Разметка согласована с корпусом.")
        sys.exit(1 if problems else 0)
    elif args.cmd == "suggest-entities":
        for name, n in suggest_entities(_chunks(cfg)):
            print(f"{n:4d}  {name}")
    elif args.cmd == "eval":
        items = load_golden(args.golden)
        asst = Assistant.from_config(cfg)
        results = evaluate_end_to_end(asst, items)
        abl = None if args.no_ablation else retrieval_ablation(cfg, items, k=asst.k)
        meta = {"retriever": asst.retriever.name, "generator": asst.generator.name, "k": asst.k,
                "chunks": len(asst.chunks)}
        path = write_report(results, args.out, meta=meta, ablation=abl)
        print(json.dumps(summarize(results), ensure_ascii=False, indent=2))
        print(f"\nОтчёт: {path}")


if __name__ == "__main__":
    main()
