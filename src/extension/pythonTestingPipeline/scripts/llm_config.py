#!/usr/bin/env python3
"""LLM client — Ollama (primary) + Groq (fallback)."""

import os
import threading
import time
from pathlib import Path

import groq
import ollama

_env = Path(__file__).parent / ".env"
if _env.exists():
    for line in _env.read_text().splitlines():
        if line.strip() and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ[k.strip()] = v.strip()

_OLLAMA_PRIORITY = [
    "minimax-m2.7:cloud",
    "qwen3-coder-next:cloud",
    "gpt-oss:120b-cloud",
]

_GROQ_FALLBACK = [
    "moonshotai/kimi-k2-instruct",
    "meta-llama/llama-4-maverick-17b-128e-instruct",
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "moonshotai/kimi-k2-instruct-0905",
    "groq/compound",
    "groq/compound-mini",
]

# (context_tokens, max_output_tokens, rpm, tpm)
MODEL_SPECS: dict[str, tuple[int, int, int, int]] = {
    "minimax-m2.7:cloud":                            (200_000, 32_768, 1000, 1_000_000),
    "qwen3-coder-next:cloud":                        (262_144, 32_768, 1000, 1_000_000),
    "gpt-oss:120b-cloud":                            (131_072, 32_768, 1000, 1_000_000),
    "moonshotai/kimi-k2-instruct":                   (131_072,  8_192,   60,    10_000),
    "moonshotai/kimi-k2-instruct-0905":              (262_144, 16_384,   60,    10_000),
    "meta-llama/llama-4-maverick-17b-128e-instruct": (131_072,  8_192, 1000,   250_000),
    "meta-llama/llama-4-scout-17b-16e-instruct":     (131_072,  8_192, 1000,   250_000),
    "groq/compound":                                 (131_072,  8_192,  200,   200_000),
    "groq/compound-mini":                            (131_072,  8_192,  200,   200_000),
}

MODELS: list[str] = list(_OLLAMA_PRIORITY) + list(_GROQ_FALLBACK)


class LLMClient:
    _cooldowns: dict[tuple[str, str], float] = {}
    _lock = threading.Lock()

    def __init__(self, **_):
        self.api_keys = [v for k, v in sorted(os.environ.items()) if k.startswith("GROQ_API_KEY")]
        self.key_idx = 0
        self._groq_client = self._make_groq_client()
        self._ollama_client = self._make_ollama_client()
        self.last_used_model: str | None = None
        self._ollama_models: set[str] = set(_OLLAMA_PRIORITY)
        self._discover_ollama_models()

    def _make_ollama_client(self):
        url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434").removesuffix("/v1")
        try:
            return ollama.Client(host=url)
        except Exception as e:
            print(f"⚠️  Ollama init failed: {e}")
            return None

    def _make_groq_client(self):
        if self.api_keys:
            return groq.Groq(api_key=self.api_keys[self.key_idx % len(self.api_keys)], max_retries=0, timeout=90.0)
        return None

    def _discover_ollama_models(self):
        if not self._ollama_client:
            return
        try:
            discovered = [m.model for m in self._ollama_client.list().models]
            insert_pos = len(_OLLAMA_PRIORITY)
            for name in discovered:
                if name not in self._ollama_models:
                    MODEL_SPECS.setdefault(name, (131_072, 8_192, 1000, 250_000))
                    if name not in MODELS:
                        MODELS.insert(insert_pos, name)
                        insert_pos += 1
                    self._ollama_models.add(name)
            print(f"   🦙 Ollama: {sorted(self._ollama_models)}")
        except Exception as e:
            print(f"⚠️  Ollama discovery failed: {e}")

    def _is_ollama(self, model: str) -> bool:
        return model in self._ollama_models

    def _cooldown_key(self, model: str) -> tuple[str, str]:
        key = "" if self._is_ollama(model) else (self.api_keys[self.key_idx][:12] if self.api_keys else "")
        return (key, model)

    def _set_cooldown(self, model: str, seconds: float):
        with self._lock:
            self._cooldowns[self._cooldown_key(model)] = time.time() + seconds

    def _is_ready(self, model: str) -> bool:
        with self._lock:
            return time.time() >= self._cooldowns.get(self._cooldown_key(model), 0)

    def _next_available(self) -> str | None:
        return next((m for m in MODELS if self._is_ready(m)), None)

    @property
    def current_model(self) -> str:
        return self._next_available() or MODELS[0]

    @property
    def current_api_key(self) -> str | None:
        k = self.api_keys[self.key_idx] if self.api_keys else None
        return f"{k[:8]}...{k[-4:]}" if k else None

    def call(self, sys_p: str, usr_p: str, temp: float = 0.2) -> tuple[str, bool]:
        tokens = len(sys_p + usr_p) // 4

        for _ in range(20):
            model = self._next_available()

            if not model:
                for _ in range(len(self.api_keys)):
                    self.key_idx = (self.key_idx + 1) % max(len(self.api_keys), 1)
                    self._groq_client = self._make_groq_client()
                    model = self._next_available()
                    if model:
                        break
                if not model:
                    print("   ⏳ All models on cooldown. Waiting 10s...")
                    time.sleep(10)
                    continue

            ctx, max_out, _, _ = MODEL_SPECS[model]
            if tokens > ctx * 0.9:
                print(f"   ⚠️  Input too long for [{model}]. Skipping.")
                self._set_cooldown(model, 60)
                continue

            is_ollama = self._is_ollama(model)
            try:
                if is_ollama:
                    resp = self._ollama_client.chat(
                        model=model,
                        messages=[{"role": "system", "content": sys_p}, {"role": "user", "content": usr_p}],
                        options={"temperature": temp, "num_predict": min(max_out, 32_768)},
                    )
                    text = resp["message"]["content"]
                else:
                    resp = self._groq_client.chat.completions.create(
                        model=model,
                        messages=[{"role": "system", "content": sys_p}, {"role": "user", "content": usr_p}],
                        temperature=temp,
                        max_tokens=min(max_out, 8_192),
                    )
                    text = resp.choices[0].message.content

                self.last_used_model = model
                return text, False

            except Exception as e:
                if isinstance(e, groq.RateLimitError):
                    cd = min(float(e.response.headers.get("retry-after", 60)) if e.response else 60, 120)
                    print(f"   ⚠️  Rate limit [{model}]: {cd:.0f}s")
                elif isinstance(e, groq.APIStatusError):
                    cd = 300 if e.status_code == 413 else 30
                    print(f"   ⚠️  Groq {e.status_code} [{model}]")
                else:
                    cd = 60 if is_ollama else 15
                    print(f"   ⚠️  Error [{model}]: {e}")
                self._set_cooldown(model, cd)

        raise RuntimeError("Exhausted all LLM attempts")


def create_llm_client(**kw) -> LLMClient:
    return LLMClient(**kw)
