"""hivest/shared/llm_client.py"""

from __future__ import annotations
import os
import time
import random
import requests
from typing import Optional, Callable, Any, Dict


# Default model name; do not read env here to avoid poisoning defaults
MODEL_DEFAULT = "llama3:8b"


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
    # Read env and sanitize; fallback to constant default
    val = os.getenv("OLLAMA_MODEL", "")
    m = _sanitize_model(val)
    return m or MODEL_DEFAULT


def _sanitize_model(m: Optional[str]) -> str:
    s = (m or "").strip()
    # strip stray quotes/backticks from env-injected values
    if s.startswith("'") and s.endswith("'") and len(s) >= 2:
        s = s[1:-1].strip()
    if s.startswith('"') and s.endswith('"') and len(s) >= 2:
        s = s[1:-1].strip()
    if s.startswith("`") and s.endswith("`") and len(s) >= 2:
        s = s[1:-1].strip()

    # guard against accidental literal env expressions or placeholders
    bad = (
        "os.getenv(",
        "${",
        "%%",
    )
    if (not s) or any(b in s for b in bad) or s.lower() in ("none", "null"):
        return MODEL_DEFAULT
    return s


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


def _post_generate(host: str, timeout: int, payload: dict, retries: int = 2, backoff: float = 1.0) -> dict:
    last_ex = None
    last_status = None
    last_text = None
    for attempt in range(retries + 1):
        try:
            r = requests.post(f"{host}/api/generate", json=payload, timeout=timeout)
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
    raise RuntimeError(f"Ollama generate failed after retries: {last_ex} | {detail}")


# hivest/shared/llm_client.py

def make_llm(model_name: str | None = None, host: str | None = None):
    raw_model = model_name if model_name is not None else _get_ollama_model()
    model = _sanitize_model(raw_model)
    base = (host or _get_ollama_host()).rstrip("/")
    timeout = _get_ollama_timeout()
    temperature = _get_ollama_temperature()

    def _call(prompt: str) -> str:
        system_msg = "You are a helpful financial analyst."
        chat_payload = {
            "model": model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt},
            ],
            "options": {
                "temperature": temperature,
                "num_predict": 5000,
            }
        }
        # Hide DeepSeek-R1 thinking if used
        if "deepseek-r1" in model.lower():
            chat_payload["options"]["stop"] = ["</think>", "<think>", "Reasoning:", "Thought:"]

        try:
            data = _post_chat(base, timeout, chat_payload, retries=2, backoff=1.0)
            return (data.get("message", {}) or {}).get("content", "").strip()
        except Exception as ex:
            # Fallback to /api/generate for older Ollama versions or edge-case errors
            gen_payload = {
                "model": model,
                "stream": False,
                "prompt": f"{system_msg}\n\n{prompt}",
                "options": chat_payload["options"],
            }
            try:
                data2 = _post_generate(base, timeout, gen_payload, retries=2, backoff=1.0)
                return (data2.get("response") or "").strip()
            except Exception:
                # Re-raise the original exception context for clearer error message upstream
                raise ex

    return _call
