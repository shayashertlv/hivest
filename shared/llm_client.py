"""Lightweight, self-contained Ollama client for Hivest (shared layer).
- No dependency on netvest; reads environment variables directly.
- Returns a callable llm(prompt: str) -> str, or None if Ollama isn't available.
"""
from __future__ import annotations
import os
import time
import random
import requests
from typing import Optional, Callable, Any, Dict


# Default model updated to llama3:8b (configurable via .env)
MODEL_DEFAULT = os.getenv("OLLAMA_MODEL", "llama3:8b").strip() or "llama3:8b"


def _get_ollama_host() -> str:
    default = "http://127.0.0.1:11434"
    val = (os.getenv("OLLAMA_HOST", default) or default).strip()
    low = val.lower()
    if not (low.startswith("http://") or low.startswith("https://")):
        return default
    if low.startswith("http://localhost") or low.startswith("https://localhost"):
        # normalize localhost to 127.0.0.1 to avoid Windows IPv6 issues
        scheme = "https" if low.startswith("https://") else "http"
        host_and_port = val.split("//", 1)[-1]
        parts = host_and_port.split(":", 1)
        port = parts[1] if len(parts) > 1 else "11434"
        return f"{scheme}://127.0.0.1:{port}"
    return val


def _get_ollama_model() -> str:
    val = (os.getenv("OLLAMA_MODEL", "") or "").strip()
    if not val or val.lower().startswith("os.getenv("):
        return MODEL_DEFAULT
    return val


def _get_ollama_timeout() -> int:
    val = os.getenv("OLLAMA_TIMEOUT", "180")
    try:
        return int(val)
    except Exception:
        return 180


def _get_ollama_temperature() -> float:
    val = os.getenv("OLLAMA_TEMPERATURE", "0.4")
    try:
        return float(val)
    except Exception:
        return 0.4


def _post_chat(host: str, timeout: int, payload: dict, retries: int = 2, backoff: float = 1.0) -> dict:
    last_ex = None
    last_status = None
    last_text = None
    for attempt in range(retries + 1):
        try:
            r = requests.post(f"{host}/api/chat", json=payload, timeout=timeout)
            last_status = r.status_code
            try:
                last_text = r.text
            except Exception:
                last_text = None
            r.raise_for_status()
            return r.json()
        except Exception as ex:
            last_ex = ex
            if attempt < retries:
                sleep_for = backoff * (2 ** attempt) + random.uniform(0.1, 0.3)
                try:
                    time.sleep(sleep_for)
                except Exception:
                    time.sleep(backoff)
    detail = f"host={host}, timeout={timeout}s, status={last_status}, body={last_text[:500] if isinstance(last_text, str) else last_text}"
    raise RuntimeError(f"Ollama chat failed after retries: {last_ex} | {detail}")


# hivest/shared/llm_client.py

def make_llm(model_name: str | None = None, host: str | None = None):
    model = model_name or _get_ollama_model()
    base = (host or _get_ollama_host()).rstrip("/")
    timeout = _get_ollama_timeout()
    temperature = _get_ollama_temperature()

    def _call(prompt: str) -> str:
        payload = {
            "model": model,
            "stream": False,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful financial analyst."
                },
                {"role": "user", "content": prompt},
            ],
            "options": {
                "temperature": temperature,
                "num_predict": 1024,
            }
        }
        # Hide DeepSeek-R1 thinking if used
        if "deepseek-r1" in model.lower():
            payload["options"]["stop"] = ["</think>", "<think>", "Reasoning:", "Thought:"]

        data = _post_chat(base, timeout, payload, retries=2, backoff=1.0)
        return (data.get("message", {}) or {}).get("content", "").strip()

    return _call
