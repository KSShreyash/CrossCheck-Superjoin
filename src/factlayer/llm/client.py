import json
import re
import sqlite3
import time

from . import cache


class NoAPIKey(RuntimeError):
    """Cache miss with no API key configured."""


class BadModelJSON(RuntimeError):
    """Model returned something that is not usable JSON."""


# free-tier quota and transient server errors are worth waiting out;
# a malformed response is not, because temperature 0 reproduces it
_RETRYABLE = ("429", "rate limit", "resource_exhausted", "quota", "exhausted",
              "503", "unavailable", "500", "internal", "deadline")


def _is_retryable(exc: Exception) -> bool:
    if isinstance(exc, (BadModelJSON, NoAPIKey)):
        return False
    blob = f"{type(exc).__name__} {exc}".lower()
    return any(marker in blob for marker in _RETRYABLE)


def _loads_lenient(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
        raise BadModelJSON(
            f"could not parse model output ({len(text)} chars); "
            "most likely truncated at the output token limit")


class LLMClient:
    def __init__(self, conn: sqlite3.Connection, api_key: str | None, model: str):
        self.conn, self.api_key, self.model = conn, api_key, model
        self._model_obj = None

    def _provider(self):
        if self._model_obj is None:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self._model_obj = genai.GenerativeModel(self.model)
        return self._model_obj

    def complete_json(self, prompt: str, prompt_version: str,
                      max_attempts: int = 5) -> dict:
        key = cache.cache_key(self.model, prompt_version, prompt)
        hit = cache.get(self.conn, key)
        if hit is not None:
            return hit
        if not self.api_key:
            raise NoAPIKey(
                "No cached response and GEMINI_API_KEY is unset. "
                "Set a key to ingest documents the cache has not seen.")

        resp = None
        for attempt in range(max_attempts):
            try:
                resp = self._provider().generate_content(
                    prompt,
                    generation_config={
                        "temperature": 0,
                        "response_mime_type": "application/json",
                        # a 12k-char window can yield a lot of facts; the
                        # default ceiling truncates the JSON mid-object
                        "max_output_tokens": 8192,
                    })
                break
            except Exception as exc:
                if attempt == max_attempts - 1 or not _is_retryable(exc):
                    raise
                # free tier rations requests per minute, so wait it out
                time.sleep(min(2 ** attempt * 2, 60))

        data = _loads_lenient(resp.text)      # BadModelJSON is not retried
        cache.put(self.conn, key, data)
        return data
