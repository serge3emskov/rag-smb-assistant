"""Загрузка документов: HTML (сохранённые из браузера страницы), PDF, Markdown/TXT.

Каждый документ разбивается на секции по заголовкам — это нужно, чтобы чанки не
смешивали разные продукты/тарифы и чтобы в цитате был виден путь «страница → раздел».
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

from bs4 import BeautifulSoup

_NOISE_TAGS = ["script", "style", "noscript", "svg", "nav", "footer", "header", "form", "iframe", "button"]
_NOISE_CLASS_RE = re.compile(r"cookie|banner|breadcrumb|footer|header|menu|modal|popup|subscribe|social", re.I)


@dataclass
class Section:
    heading: str          # "Страница > Раздел > Подраздел"
    text: str


@dataclass
class Document:
    doc_id: str
    title: str
    source: str           # путь или URL
    sections: list[Section] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n\n".join(f"{s.heading}\n{s.text}" for s in self.sections)


def _doc_id(path: Path) -> str:
    stem = re.sub(r"[^a-z0-9а-я_-]+", "_", path.stem.lower()).strip("_")
    return stem or hashlib.md5(str(path).encode()).hexdigest()[:10]


def _clean(text: str) -> str:
    text = text.replace(" ", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n", text)
    return text.strip()


# ---------------------------------------------------------------- HTML
def load_html(path: Path) -> Document:
    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="ignore"), "html.parser")
    title = (soup.title.get_text(strip=True) if soup.title else path.stem)
    canonical = soup.find("link", rel="canonical")
    source = canonical["href"] if canonical and canonical.get("href") else str(path)

    for tag in soup(_NOISE_TAGS):
        tag.decompose()
    for tag in soup.find_all(attrs={"class": _NOISE_CLASS_RE}):
        tag.decompose()

    root = soup.find("main") or soup.body or soup
    sections: list[Section] = []
    stack: list[tuple[int, str]] = []
    buf: list[str] = []

    def flush():
        txt = _clean("\n".join(buf))
        if txt:
            path_ = " > ".join([title] + [h for _, h in stack])
            sections.append(Section(path_, txt))
        buf.clear()

    for el in root.find_all(["h1", "h2", "h3", "h4", "p", "li", "td", "th", "div", "span"]):
        # берём текст только «листовых» блоков, чтобы не дублировать вложенное
        if el.name in ("div", "span") and el.find(["p", "li", "div", "h1", "h2", "h3", "h4", "table"]):
            continue
        txt = el.get_text(" ", strip=True)
        if not txt:
            continue
        if el.name in ("h1", "h2", "h3", "h4"):
            flush()
            level = int(el.name[1])
            stack = [(lv, h) for lv, h in stack if lv < level]
            if txt != title:
                stack.append((level, txt))
        else:
            if not buf or buf[-1] != txt:
                buf.append(("• " if el.name == "li" else "") + txt)
    flush()
    return Document(_doc_id(path), title, source, _dedupe(sections))


def _dedupe(sections: list[Section]) -> list[Section]:
    seen, out = set(), []
    for s in sections:
        key = hashlib.md5(s.text.encode()).hexdigest()
        if key not in seen and len(s.text) > 20:
            seen.add(key)
            out.append(s)
    return out


# ---------------------------------------------------------------- PDF
def load_pdf(path: Path) -> Document:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    sections = []
    for i, page in enumerate(reader.pages, 1):
        txt = _clean(page.extract_text() or "")
        if txt:
            sections.append(Section(f"{path.stem} > стр. {i}", txt))
    title = (reader.metadata.title if reader.metadata and reader.metadata.title else path.stem)
    return Document(_doc_id(path), title, str(path), sections)


# ---------------------------------------------------------------- Markdown / TXT
def load_markdown(path: Path) -> Document:
    raw = path.read_text(encoding="utf-8")
    source = str(path)
    m = re.match(r"^---\n(.*?)\n---\n", raw, re.S)          # простой front-matter
    if m:
        for line in m.group(1).splitlines():
            if line.startswith("source:"):
                source = line.split(":", 1)[1].strip()
        raw = raw[m.end():]

    title, sections, stack, buf = path.stem, [], [], []

    def flush():
        txt = _clean("\n".join(buf))
        if txt:
            sections.append(Section(" > ".join([title] + [h for _, h in stack]), txt))
        buf.clear()

    for line in raw.splitlines():
        h = re.match(r"^(#{1,4})\s+(.*)", line)
        if h:
            level, text = len(h.group(1)), h.group(2).strip()
            if level == 1:
                flush()
                title, stack = text, []
                continue
            flush()
            stack = [(lv, t) for lv, t in stack if lv < level]
            stack.append((level, text))
        else:
            buf.append(line)
    flush()
    return Document(_doc_id(path), title, source, sections)


LOADERS = {".html": load_html, ".htm": load_html, ".pdf": load_pdf, ".md": load_markdown, ".txt": load_markdown}


def load_dir(folder: str | Path) -> list[Document]:
    folder = Path(folder)
    docs = []
    for path in sorted(folder.rglob("*")):
        loader = LOADERS.get(path.suffix.lower())
        if loader and path.is_file():
            doc = loader(path)
            if doc.sections:
                docs.append(doc)
    if not docs:
        raise FileNotFoundError(f"В {folder} нет документов ({', '.join(LOADERS)})")
    return docs
