import json
import re
import sqlite3
import time

from . import cache


class NoAPIKey(RuntimeError):
    """Cache miss with no API key configured."""


class BadModelJSON(RuntimeError):
    """Model returned something that is not usable JSON."""


class DailyQuotaExhausted(RuntimeError):
    """This model's daily allowance is gone; waiting will not help."""


# server-side hiccups: worth waiting out, and they cost no allowance
_RETRYABLE = ("503", "unavailable", "500", "internal", "deadline",
              "timeout", "connection")

# anything that means "you have had your share"
_QUOTA = ("429", "rate limit", "resource_exhausted", "resourceexhausted",
          "quota", "exhausted")


def _is_quota(exc: Exception) -> bool:
    """Any refusal on grounds of allowance, per-minute or per-day alike."""
    if isinstance(exc, (BadModelJSON, NoAPIKey, DailyQuotaExhausted)):
        return False
    blob = f"{type(exc).__name__} {exc}".lower()
    return any(marker in blob for marker in _QUOTA)


def _is_retryable(exc: Exception) -> bool:
    if isinstance(exc, (BadModelJSON, NoAPIKey, DailyQuotaExhausted)):
        return False
    if _is_quota(exc):
        return False        # rotate instead of sleeping
    blob = f"{type(exc).__name__} {exc}".lower()
    return any(marker in blob for marker in _RETRYABLE)


def _unwrap(data):
    """Some models wrap the object in a single-element array; unwrap it."""
    if isinstance(data, list):
        objects = [x for x in data if isinstance(x, dict)]
        if len(objects) == 1:
            return objects[0]
        if objects:
            # several partial objects: merge their list-valued keys
            merged: dict = {}
            for obj in objects:
                for k, v in obj.items():
                    if isinstance(v, list):
                        merged.setdefault(k, []).extend(v)
                    else:
                        merged.setdefault(k, v)
            return merged
    return data


def _scan_values(text: str) -> list:
    """Every complete JSON value in the text, in order."""
    decoder = json.JSONDecoder()
    values, i, n = [], 0, len(text)
    while i < n:
        ch = text[i]
        if ch not in "[{":
            i += 1
            continue
        try:
            value, end = decoder.raw_decode(text, i)
        except json.JSONDecodeError:
            i += 1
            continue
        values.append(value)
        i = end
    return values


def _is_envelope(value) -> bool:
    """A result of the shape the prompts ask for: {"facts": [...]} and such."""
    return isinstance(value, dict) and any(
        isinstance(v, list) for v in value.values())


def _loads_lenient(text: str) -> dict:
    try:
        return _unwrap(json.loads(text))
    except json.JSONDecodeError:
        envelopes = [v for v in _scan_values(text)
                     if _is_envelope(v) or _is_envelope(_unwrap(v))]
        if envelopes:
            return _unwrap(envelopes if len(envelopes) > 1 else envelopes[0])
        # keep a sample, or a failure is only a character count
        sample = re.sub(r"\s+", " ", text)[:240]
        raise BadModelJSON(
            f"could not parse model output ({len(text)} chars). "
            f"Starts: {sample!r}")


class LLMClient:
    """Cached Gemini client that can fall through exhausted models and keys."""

    def __init__(self, conn: sqlite3.Connection, api_key: str | None, model: str,
                 models: list[str] | None = None,
                 api_keys: list[str] | None = None):
        self.conn = conn
        self.models = [m for m in (models or [model]) if m]
        self.api_keys = [k for k in (api_keys or ([api_key] if api_key else [])) if k]
        self.api_key = self.api_keys[0] if self.api_keys else None
        self.model = self.models[0]
        self._spent: set[tuple[str, str]] = set()   # (key, model) pairs used up
        self._cache_obj = None
        self._cache_for: tuple[str, str] | None = None

    def _provider(self, key: str, model: str):
        if self._cache_for != (key, model):
            import google.generativeai as genai
            genai.configure(api_key=key)
            self._cache_obj = genai.GenerativeModel(model)
            self._cache_for = (key, model)
        return self._cache_obj

    def _combinations(self):
        """Every key and model pairing that has not been used up yet."""
        for key in self.api_keys:
            for model in self.models:
                if (key, model) not in self._spent:
                    yield key, model

    _GENERATION = {
        "temperature": 0,
        "response_mime_type": "application/json",
        # a dense window yields many facts and the default ceiling truncates them
        "max_output_tokens": 8192,
    }

    def complete_json(self, prompt: str, prompt_version: str,
                      max_attempts: int = 4) -> dict:
        # a cached answer under any rotated model is still an answer
        for model in self.models:
            hit = cache.get(self.conn,
                            cache.cache_key(model, prompt_version, prompt))
            if hit is not None:
                return hit
        if not self.api_keys:
            raise NoAPIKey(
                "No cached response and GEMINI_API_KEY is unset. "
                "Set a key to ingest documents the cache has not seen.")

        last_exc: Exception | None = None
        for api_key, model in self._combinations():
            for attempt in range(max_attempts):
                try:
                    resp = self._provider(api_key, model).generate_content(
                        prompt, generation_config=self._GENERATION)
                    data = _loads_lenient(resp.text)   # BadModelJSON not retried
                    cache.put(self.conn,
                              cache.cache_key(model, prompt_version, prompt), data)
                    self.model, self.api_key = model, api_key
                    return data
                except Exception as exc:               # noqa: BLE001
                    last_exc = exc
                    if _is_quota(exc):
                        # this pairing is done for the day; try the next one
                        self._spent.add((api_key, model))
                        break
                    if attempt == max_attempts - 1 or not _is_retryable(exc):
                        raise
                    # a per-minute limit, so wait it out
                    time.sleep(min(2 ** attempt * 2, 60))

        raise DailyQuotaExhausted(
            f"every model and key pairing is out of daily quota "
            f"({len(self.api_keys)} key(s) x {len(self.models)} model(s)). "
            f"Last error: {str(last_exc)[:160]}")
