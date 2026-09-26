"""Нормализация и токенизация русского текста."""
from __future__ import annotations

import re
from functools import lru_cache

import snowballstemmer

_STEMMER = snowballstemmer.stemmer("russian")
_TOKEN_RE = re.compile(r"[a-zа-я0-9]+(?:[.,][0-9]+)?", re.IGNORECASE)

# Короткие служебные слова, которые только шумят в BM25
STOPWORDS = frozenset(
    """
    а в во и или к ко на над о об от по под при про с со у за из до для без же ли бы
    не ни но что как это то так там тут где когда какой какая какие каков какова
    ли ль мне меня мы вы вам вас он она оно они его её ее их есть быть был была
    можно нужно ли у меня мой моя мои ваш ваша ваши я ты
    """.split()
)


def normalize(text: str) -> str:
    """Нижний регистр, ё→е, унификация пробелов и чисел (1 000 → 1000)."""
    text = text.lower().replace("ё", "е")
    text = re.sub(r"(?<=\d)[\s  ](?=\d{3}\b)", "", text)
    text = text.replace(" ", " ")
    return re.sub(r"\s+", " ", text).strip()


@lru_cache(maxsize=200_000)
def stem(word: str) -> str:
    return _STEMMER.stemWord(word)


def tokenize(text: str, *, stem_words: bool = True, drop_stop: bool = True) -> list[str]:
    tokens = _TOKEN_RE.findall(normalize(text))
    if drop_stop:
        tokens = [t for t in tokens if t not in STOPWORDS]
    if stem_words:
        tokens = [stem(t) if not t[0].isdigit() else t for t in tokens]
    return tokens


_SENT_RE = re.compile(r"(?<=[.!?…])\s+(?=[A-ZА-ЯЁ0-9«\"(])|\n+")


def split_sentences(text: str) -> list[str]:
    parts = [s.strip(" -•\t") for s in _SENT_RE.split(text)]
    return [p for p in parts if len(p) > 2]


_NUM_RE = re.compile(r"\d+(?:[.,]\d+)?")


def numbers(text: str) -> set[str]:
    """Все числа из текста в каноническом виде (для проверки галлюцинаций в цифрах)."""
    return {n.replace(",", ".") for n in _NUM_RE.findall(normalize(text))}
