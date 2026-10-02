#!/usr/bin/env python3
"""strategies/jev/deciders.py — the swappable model behind the jev strategy.

Every backend implements one interface, `Decider.decide(state, question)`, and
returns the same validated answer:

    {"choice": <one of question["criteria"]>, "probabilities": {option: p in [0,1]},
     "confidence": float | None, "reason": str, "model": resolved model id}

Pick one with a "backend:model" spec (config.JEV_DECIDER, or JEV_DECIDER in .env):

    jev:jev-latest                       TypeSafe's hosted Jev (System One)
    mlx:mlx-community/Qwen3-0.6B-4bit    local model in-process on Apple silicon
    openai:qwen3:4b                      any OpenAI-compatible chat server

Add a backend = subclass Decider (or ChatDecider for a chat LLM), implement
`_decide` (or `_chat`), optionally `check()` / `_cache_extra()`, and register it
in DECIDERS.

Answers are validated, then cached on disk keyed by everything that shapes them
(backend, model, state, question, plus each backend's prompt/settings), so a
re-run backtest over the same bars makes no model calls — and a changed prompt or
setting never silently reuses old answers. Any failure is a DeciderError (the bot
then stays flat for that bar); configuration problems surface at startup through
check().
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout

import requests

import config
from logsetup import get_logger

log = get_logger()


class DeciderError(RuntimeError):
    """The decider could not answer (no key, HTTP error, timeout, bad reply)."""


class Decider(ABC):
    backend = ""

    def __init__(self, model: str, cache_path=None):
        self.model = model
        self.cache_path = config.JEV_CACHE_PATH if cache_path is None else cache_path
        self._cache = None                  # lazy: {request hash: answer}
        self._resolved = None               # model version the backend reported

    @property
    def spec(self) -> str:
        return f"{self.backend}:{self.model}"

    def check(self):
        """Validate configuration at startup (raise SystemExit if unusable)."""

    def decide(self, state: dict, question: dict) -> dict:
        req = {"backend": self.backend, "model": self.model, "state": state,
               "question": question, "extra": self._cache_extra(question)}
        key = hashlib.sha256(json.dumps(req, sort_keys=True).encode()).hexdigest()
        cache = self._load_cache()
        if key in cache:
            return cache[key]
        try:
            raw = self._decide(state, question)
        except DeciderError:
            raise
        except Exception as e:              # never let a backend crash the bot loop
            raise DeciderError(f"{type(e).__name__}: {e}") from e
        ans = validate(raw, list(question["criteria"]))
        resolved = ans.get("model")
        if resolved and resolved != self._resolved:
            log.info("jev decider %s resolved to model %s", self.spec, resolved)
            self._resolved = resolved
        cache[key] = ans
        if self.cache_path:
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
            with open(self.cache_path, "a") as f:
                f.write(json.dumps({"key": key, "answer": ans}) + "\n")
        return ans

    @abstractmethod
    def _decide(self, state: dict, question: dict) -> dict:
        """Answer one choice question (uncached, unvalidated)."""

    def _cache_extra(self, question: dict) -> dict:
        """Backend settings that change the answer (part of the cache key)."""
        return {}

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


def validate(raw, options) -> dict:
    """The standard answer, or DeciderError: `choice` must be one of the offered
    options (case-insensitive); probabilities are finite and clamped to [0, 1]."""
    if not isinstance(raw, dict) or not isinstance(raw.get("choice"), str):
        raise DeciderError(f"malformed answer: {str(raw)[:200]}")
    choice = raw["choice"].strip().lower()
    if choice not in options:
        raise DeciderError(f"chose {choice!r}, not one of {options}")
    probs = raw.get("probabilities")
    if not isinstance(probs, dict):
        probs = {}
    clean = {}
    for k, v in probs.items():
        try:
            v = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(v):
            clean[str(k).strip().lower()] = min(1.0, max(0.0, v))
    conf = raw.get("confidence")
    try:
        conf = min(1.0, max(0.0, float(conf))) if conf is not None else None
        if conf is not None and not math.isfinite(conf):
            conf = None
    except (TypeError, ValueError):
        conf = None
    return {"choice": choice, "probabilities": clean, "confidence": conf,
            "reason": str(raw.get("reason") or "")[:300],
            "model": raw.get("model")}


def _post(url, body, headers=None, timeout=None, retries=None) -> dict:
    """POST JSON → JSON. Retries rate limits / overload / 5xx / connection errors
    with exponential backoff; anything else (or retries exhausted) is a
    DeciderError."""
    retries = config.JEV_RETRIES if retries is None else retries
    for attempt in range(retries + 1):
        try:
            r = requests.post(url, json=body, headers=headers or {},
                              timeout=timeout or config.JEV_TIMEOUT)
        except requests.RequestException as e:
            if attempt < retries and isinstance(e, (requests.ConnectionError,
                                                    requests.Timeout)):
                time.sleep(0.5 * 2 ** attempt)
                continue
            raise DeciderError(f"request failed: {e}") from e
        if r.status_code in (429, 500, 502, 503, 504, 529) and attempt < retries:
            time.sleep(0.5 * 2 ** attempt)
            continue
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

    def check(self):
        if not self.api_key:
            raise SystemExit("JEV_DECIDER uses TypeSafe Jev but TYPESAFE_AI_API_KEY "
                             "is not set (add it to .env)")

    def _decide(self, state, question):
        if not self.api_key:
            raise DeciderError("TYPESAFE_AI_API_KEY is not set (add it to .env)")
        data = _post(f"{self.base}/systemone",
                     {"model": self.model, "state": state,
                      "questions": {"pick": question}},
                     {"Authorization": f"Bearer {self.api_key}"},
                     timeout=config.JEV_API_TIMEOUT)
        try:
            a = data["answers"]["pick"]
        except (KeyError, TypeError) as e:
            raise DeciderError(f"malformed reply: {str(data)[:200]}") from e
        if not isinstance(a, dict):
            raise DeciderError(f"malformed reply: {str(data)[:200]}")
        return {**a, "model": data.get("model")}


# ── chat LLMs (local or served) ─────────────────────────────────────────────

class ChatDecider(Decider):
    """A chat/reasoning LLM: the question becomes a system prompt, the state the
    user message, and the model replies with JSON {choice, confidence, reason}
    (after reasoning, if it thinks)."""

    @abstractmethod
    def _chat(self, system: str, user: str) -> str:
        """One chat completion → the raw reply text."""

    def _cache_extra(self, question):
        return {"prompt": system_prompt(question), "think": config.JEV_LLM_THINK,
                "max_tokens": config.JEV_LLM_MAX_TOKENS}

    def _decide(self, state, question):
        options = list(question["criteria"])
        text = self._chat(system_prompt(question), json.dumps(state))
        return {**parse_reply(text, options), "model": self.model}


class MlxDecider(ChatDecider):
    """A local model run in-process with mlx-lm (Apple silicon), loaded once at
    startup. Each answer has a hard JEV_TIMEOUT deadline. `model` is a Hugging
    Face repo id or a local path."""
    backend = "mlx"

    def __init__(self, model="mlx-community/Qwen3-0.6B-4bit", cache_path=None):
        super().__init__(model, cache_path)
        self._lm = None
        self._pool = ThreadPoolExecutor(max_workers=1)
        self._busy = None                   # a generation still running past its deadline

    def check(self):
        try:
            import mlx_lm  # noqa: F401
        except ImportError as e:
            raise SystemExit("JEV_DECIDER uses the mlx backend, which needs mlx-lm "
                             "(pip install mlx-lm, ideally in its own venv)") from e
        self._load()

    def _load(self):
        if self._lm is None:
            from mlx_lm import load
            self._lm = load(self.model)
        return self._lm

    def _generate(self, system, user):
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler
        model, tok = self._load()
        prompt = tok.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            add_generation_prompt=True, tokenize=False,
            enable_thinking=config.JEV_LLM_THINK)
        return generate(model, tok, prompt=prompt, verbose=False,
                        max_tokens=config.JEV_LLM_MAX_TOKENS,
                        sampler=make_sampler(temp=0.0))

    def _chat(self, system, user):
        try:
            import mlx_lm  # noqa: F401
        except ImportError as e:
            raise DeciderError("mlx-lm is not installed") from e
        if self._busy is not None and not self._busy.done():
            raise DeciderError("previous generation still running")
        fut = self._pool.submit(self._generate, system, user)
        try:
            return fut.result(timeout=config.JEV_TIMEOUT)
        except FutureTimeout as e:
            self._busy = fut
            raise DeciderError(f"no answer within {config.JEV_TIMEOUT}s") from e


class OpenAIChatDecider(ChatDecider):
    """Any OpenAI-compatible /chat/completions server at JEV_OPENAI_URL —
    Ollama (default URL), LM Studio, mlx_lm.server, llama.cpp, vLLM."""
    backend = "openai"

    def __init__(self, model="qwen3:4b", url=None, api_key=None, cache_path=None):
        super().__init__(model, cache_path)
        self.url = (url or config.JEV_OPENAI_URL).rstrip("/")
        self.api_key = config.JEV_OPENAI_KEY if api_key is None else api_key

    def _cache_extra(self, question):
        return {**super()._cache_extra(question), "url": self.url}

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
    lines = [ins["question"], "", "Goal: " + ins["goal"]]
    if ins.get("timing"):
        lines += ["", "Timing: " + ins["timing"]]
    lines += ["", "Inputs: " + ins["inputs"]]
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


def parse_reply(text: str, options) -> dict:
    """The last JSON object with a valid `choice` in a chat reply (any <think>
    block stripped; nested objects and braces inside strings are fine) → the
    standard answer, with probabilities spread from the stated confidence."""
    body = _THINK.sub("", text or "")
    dec = json.JSONDecoder()
    found = None
    for m in re.finditer(r"\{", body):
        try:
            obj, _end = dec.raw_decode(body, m.start())
        except ValueError:
            continue
        if isinstance(obj, dict) and \
                str(obj.get("choice", "")).strip().lower() in options:
            found = obj
    if found is None:
        raise DeciderError(f"no valid answer in reply: {body[-200:]!r}")
    choice = str(found["choice"]).strip().lower()
    try:
        conf = min(1.0, max(0.0, float(found.get("confidence", 0.5))))
        if not math.isfinite(conf):
            conf = 0.5
    except (TypeError, ValueError):
        conf = 0.5
    rest = (1.0 - conf) / max(1, len(options) - 1)
    return {"choice": choice,
            "probabilities": {o: round(conf if o == choice else rest, 2)
                              for o in options},
            "confidence": conf, "reason": str(found.get("reason", ""))[:300]}


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
