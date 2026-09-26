"""Бэкенды генерации.

* QwenHF        — Qwen через transformers (Colab / локальный GPU);
* OpenAICompat  — любой OpenAI-совместимый сервер: Ollama (qwen3:4b), vLLM, LM Studio, облачные API;
* ExtractiveBaseline — без LLM: выбирает предложения из контекста. Нужен для офлайн-тестов
  пайплайна и как нижняя граница качества («что даёт LLM сверх простого извлечения»).
"""
from __future__ import annotations

import re
from typing import Protocol

from .prompts import REFUSAL, build_messages
from .text import split_sentences, tokenize

_THINK_RE = re.compile(r"<think>.*?</think>", re.S)


class Generator(Protocol):
    name: str

    def generate(self, question: str, hits) -> str: ...

    def chat(self, messages: list[dict], max_new_tokens: int = 512) -> str: ...


class QwenHF:
    def __init__(self, model_name: str = "Qwen/Qwen3-4B-Instruct-2507", max_new_tokens: int = 400,
                 load_in_4bit: bool = False, device_map: str = "auto"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name = model_name.split("/")[-1]
        self.tok = AutoTokenizer.from_pretrained(model_name)
        kw = {"device_map": device_map}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                                                           bnb_4bit_quant_type="nf4")
        else:
            kw["torch_dtype"] = torch.float16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **kw)
        self.model.eval()
        self.max_new_tokens = max_new_tokens

    def chat(self, messages: list[dict], max_new_tokens: int | None = None) -> str:
        import torch

        try:   # у гибридных Qwen3 отключаем режим рассуждений
            prompt = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                  enable_thinking=False)
        except TypeError:
            prompt = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tok(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens or self.max_new_tokens,
                                      do_sample=False, temperature=None, top_p=None, top_k=None,
                                      repetition_penalty=1.05, pad_token_id=self.tok.eos_token_id)
        text = self.tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return _THINK_RE.sub("", text).strip()

    def generate(self, question: str, hits) -> str:
        return self.chat(build_messages(question, hits))


class OpenAICompat:
    def __init__(self, model: str = "qwen3:4b", base_url: str = "http://localhost:11434/v1",
                 api_key: str = "ollama", max_new_tokens: int = 400, temperature: float = 0.0):
        from openai import OpenAI

        self.name = model
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    def chat(self, messages: list[dict], max_new_tokens: int | None = None) -> str:
        r = self.client.chat.completions.create(model=self.name, messages=messages,
                                                temperature=self.temperature,
                                                max_tokens=max_new_tokens or self.max_new_tokens)
        return _THINK_RE.sub("", r.choices[0].message.content or "").strip()

    def generate(self, question: str, hits) -> str:
        return self.chat(build_messages(question, hits))


class ExtractiveBaseline:
    """Берёт 1–2 предложения с наибольшим пересечением лемм с вопросом, проставляет ссылки.
    Отказывается, если пересечение слишком мало."""

    name = "extractive-baseline"

    def __init__(self, min_overlap: float = 0.34, max_sents: int = 2):
        self.min_overlap = min_overlap
        self.max_sents = max_sents

    def generate(self, question: str, hits) -> str:
        q = set(tokenize(question))
        if not q:
            return REFUSAL
        scored = []
        for i, h in enumerate(hits, 1):
            head = set(tokenize(h.chunk.heading))
            for s in split_sentences(h.chunk.text):
                toks = set(tokenize(s))
                overlap = len(q & (toks | head)) / len(q)
                scored.append((overlap + 0.01 * len(q & toks) - 0.001 * i, s, i))
        scored.sort(key=lambda x: -x[0])
        if not scored or scored[0][0] < self.min_overlap:
            return REFUSAL
        best = [x for x in scored[: self.max_sents] if x[0] >= self.min_overlap]
        return " ".join(f"{s.rstrip('.')}. [{i}]" for _, s, i in best)

    def chat(self, messages, max_new_tokens=None) -> str:     # судья недоступен без LLM
        raise NotImplementedError("ExtractiveBaseline не умеет работать как LLM-судья")


def build_generator(cfg: dict):
    kind = cfg.get("backend", "extractive")
    if kind == "hf":
        return QwenHF(cfg.get("model", "Qwen/Qwen3-4B-Instruct-2507"), cfg.get("max_new_tokens", 400),
                      cfg.get("load_in_4bit", False))
    if kind == "openai":
        return OpenAICompat(cfg.get("model", "qwen3:4b"), cfg.get("base_url", "http://localhost:11434/v1"),
                            cfg.get("api_key", "ollama"), cfg.get("max_new_tokens", 400))
    if kind == "extractive":
        return ExtractiveBaseline()
    raise ValueError(f"Неизвестный backend: {kind}")
