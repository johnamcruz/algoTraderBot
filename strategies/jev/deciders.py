#!/usr/bin/env python3
"""strategies/jev/deciders.py — the swappable model behind the jev strategy.

Every backend implements one interface, `Decider.decide(state, question)`, and
returns the same answer:

    {"choice": "long" | "short" | "none", "probabilities": {option: p},
     "confidence": float | None, "reason": str}

Pick one with a "backend:model" spec (config.JEV_DECIDER, or JEV_DECIDER in .env):

    jev:jev-latest                       TypeSafe's hosted Jev (System One)
    mlx:mlx-community/Qwen3-0.6B-4bit    local model in-process on Apple silicon
    openai:qwen3:4b                      any OpenAI-compatible chat server

Add a backend = subclass Decider (or ChatDecider for a chat LLM), implement
`_decide` (or `_chat`), and register it in DECIDERS.

Answers are cached on disk keyed by (backend, model, state, question), so a
re-run backtest over the same bars makes no model calls.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from abc import ABC, abstractmethod

import requests

import config


class DeciderError(RuntimeError):
    """The decider could not answer (no key, HTTP error, timeout, bad reply)."""


class Decider(ABC):
    backend = ""

    def __init__(self, model: str, cache_path=None):
        self.model = model
        self.cache_path = config.JEV_CACHE_PATH if cache_path is None else cache_path
        self._cache = None                  # lazy: {request hash: answer}

    @property
    def spec(self) -> str:
        return f"{self.backend}:{self.model}"

    def decide(self, state: dict, question: dict) -> dict:
        req = {"backend": self.backend, "model": self.model,
               "state": state, "question": question}
        key = hashlib.sha256(json.dumps(req, sort_keys=True).encode()).hexdigest()
        cache = self._load_cache()
        if key in cache:
            return cache[key]
        ans = self._decide(state, question)
        cache[key] = ans
        if self.cache_path:
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
            with open(self.cache_path, "a") as f:
                f.write(json.dumps({"key": key, "answer": ans}) + "\n")
        return ans

    @abstractmethod
    def _decide(self, state: dict, question: dict) -> dict:
        """Answer one choice question (uncached)."""

    def _load_cache(self) -> dict:
        if self._cache is None:
            self._cache = {}
            if self.cache_path and os.path.exists(self.cache_path):
                with open(self.cache_path) as f:
                    for line in f:
                        try:
                            rec = json.loads(line)
                            self._cache[rec["key"]] = rec["answer"]
                        except (ValueError, KeyError):
                            continue        # skip a torn/partial line
        return self._cache


def _post(url, body, headers=None, timeout=None) -> dict:
    try:
        r = requests.post(url, json=body, headers=headers or {},
                          timeout=timeout or config.JEV_TIMEOUT)
    except requests.RequestException as e:
        raise DeciderError(f"request failed: {e}") from e
    if r.status_code != 200:
        raise DeciderError(f"HTTP {r.status_code}: {r.text[:200]}")
    try:
        return r.json()
    except ValueError as e:
        raise DeciderError(f"malformed reply: {r.text[:200]}") from e


# ── TypeSafe Jev ────────────────────────────────────────────────────────────

class JevDecider(Decider):
    """TypeSafe's hosted Jev — the wire format `@ai-sdk/typesafe-ai` uses:
    POST {JEV_API_BASE}/systemone {"model", "state", "questions": {"pick": ...}}."""
    backend = "jev"

    def __init__(self, model="jev-latest", api_key=None, base=None, cache_path=None):
        super().__init__(model, cache_path)
        self.api_key = config.TYPESAFE_AI_API_KEY if api_key is None else api_key
        self.base = (base or config.JEV_API_BASE).rstrip("/")

    def _decide(self, state, question):
        if not self.api_key:
            raise DeciderError("TYPESAFE_AI_API_KEY is not set (add it to .env)")
        data = _post(f"{self.base}/systemone",
                     {"model": self.model, "state": state,
                      "questions": {"pick": question}},
                     {"Authorization": f"Bearer {self.api_key}"})
        try:
            a = data["answers"]["pick"]
            return {"choice": a["choice"],
                    "probabilities": a.get("probabilities") or {a["choice"]: 1.0},
                    "confidence": a.get("confidence"), "reason": ""}
        except (KeyError, TypeError) as e:
            raise DeciderError(f"malformed reply: {str(data)[:200]}") from e


# ── chat LLMs (local or served) ─────────────────────────────────────────────

class ChatDecider(Decider):
    """A chat/reasoning LLM: the question becomes a system prompt, the state the
    user message, and the model replies with JSON {choice, confidence, reason}
    (after reasoning, if it thinks)."""

    @abstractmethod
    def _chat(self, system: str, user: str) -> str:
        """One chat completion → the raw reply text."""

    def _decide(self, state, question):
        options = list(question["criteria"])
        text = self._chat(system_prompt(question), json.dumps(state))
        return parse_reply(text, options)


class MlxDecider(ChatDecider):
    """A local model run in-process with mlx-lm (Apple silicon), loaded once.
    `model` is a Hugging Face repo id or a local path."""
    backend = "mlx"

    def __init__(self, model="mlx-community/Qwen3-0.6B-4bit", cache_path=None):
        super().__init__(model, cache_path)
        self._lm = None

    def _chat(self, system, user):
        try:
            from mlx_lm import generate, load
            from mlx_lm.sample_utils import make_sampler
        except ImportError as e:
            raise SystemExit("the mlx backend needs mlx-lm: pip install mlx-lm") from e
        if self._lm is None:
            self._lm = load(self.model)
        model, tok = self._lm
        prompt = tok.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            add_generation_prompt=True, tokenize=False,
            enable_thinking=config.JEV_LLM_THINK)
        return generate(model, tok, prompt=prompt, verbose=False,
                        max_tokens=config.JEV_LLM_MAX_TOKENS,
                        sampler=make_sampler(temp=0.0))


class OpenAIChatDecider(ChatDecider):
    """Any OpenAI-compatible /chat/completions server at JEV_OPENAI_URL —
    Ollama (default URL), LM Studio, mlx_lm.server, llama.cpp, vLLM."""
    backend = "openai"

    def __init__(self, model="qwen3:4b", url=None, api_key=None, cache_path=None):
        super().__init__(model, cache_path)
        self.url = (url or config.JEV_OPENAI_URL).rstrip("/")
        self.api_key = config.JEV_OPENAI_KEY if api_key is None else api_key

    def _chat(self, system, user):
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        data = _post(f"{self.url}/chat/completions", {
            "model": self.model, "temperature": 0, "seed": 0,
            "max_tokens": config.JEV_LLM_MAX_TOKENS,
            # Qwen3-style thinking toggle (mlx_lm.server, vLLM; ignored elsewhere)
            "chat_template_kwargs": {"enable_thinking": config.JEV_LLM_THINK},
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
        }, headers)
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            raise DeciderError(f"malformed reply: {str(data)[:200]}") from e


def system_prompt(question: dict) -> str:
    """The choice question rendered as instructions for a chat model."""
    ins = question["instructions"]
    lines = [ins["question"], "", "Goal: " + ins["goal"], "", "Inputs: " + ins["inputs"]]
    if ins.get("playbook"):
        lines += ["", "Strategy playbook:"]
        lines += [f"- {k}: {v}" for k, v in ins["playbook"].items()]
    lines += ["", "Options:"]
    lines += [f"- {k}: {v}" for k, v in question["criteria"].items()]
    opts = ", ".join(f'"{o}"' for o in question["criteria"])
    lines += ["", "The user message is the current market state as JSON. Reason over "
              "all of it, then end your reply with exactly one JSON object with "
              f"`choice` (exactly one of {opts}), `confidence` (0 to 1, your "
              "probability the choice is right) and `reason` (one sentence, in "
              "your own words, citing the evidence you used)."]
    return "\n".join(lines)


_THINK = re.compile(r"<think>.*?</think>", re.S)
_OBJ = re.compile(r"\{[^{}]*\}")


def parse_reply(text: str, options) -> dict:
    """The last JSON object with a valid `choice` in a chat reply (any <think>
    block stripped) → the standard answer."""
    body = _THINK.sub("", text or "")
    for raw in reversed(_OBJ.findall(body)):
        try:
            a = json.loads(raw)
        except ValueError:
            continue
        choice = str(a.get("choice", "")).strip().lower()
        if choice not in options:
            continue
        try:
            conf = min(1.0, max(0.0, float(a.get("confidence", 0.5))))
        except (TypeError, ValueError):
            conf = 0.5
        rest = (1.0 - conf) / max(1, len(options) - 1)
        return {"choice": choice,
                "probabilities": {o: round(conf if o == choice else rest, 2)
                                  for o in options},
                "confidence": conf, "reason": str(a.get("reason", ""))[:300]}
    raise DeciderError(f"no valid answer in reply: {body[-200:]!r}")


DECIDERS = {JevDecider.backend: JevDecider, MlxDecider.backend: MlxDecider,
            OpenAIChatDecider.backend: OpenAIChatDecider}


def make_decider(spec=None) -> Decider:
    """Build a decider from "backend:model" (default config.JEV_DECIDER). The
    model part may itself contain colons (e.g. openai:qwen3:4b)."""
    spec = spec or config.JEV_DECIDER
    backend, _, model = spec.partition(":")
    if backend not in DECIDERS:
        raise SystemExit(f"unknown decider backend {backend!r} in {spec!r} "
                         f"(have {', '.join(DECIDERS)})")
    return DECIDERS[backend](model) if model else DECIDERS[backend]()
