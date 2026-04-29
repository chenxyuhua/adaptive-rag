"""Google Gemini implementation of LLMClient.

Uses the unified `google-genai` SDK. Reads the API key from the GOOGLE_API_KEY
env var (free key from https://aistudio.google.com/app/apikey). Includes a
client-side rate limiter sized for Gemini's free tier and retry-with-backoff
for transient errors.
"""

from __future__ import annotations

import os
import threading
import time
from collections import deque

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ..schemas import LLMUsage
from .base import GenerationResult, LLMClient


class _RateLimiter:
    """Token-bucket-ish: cap calls to N per 60 seconds. Thread-safe."""

    def __init__(self, requests_per_minute: int):
        self.rpm = max(1, int(requests_per_minute))
        self._times: deque[float] = deque()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            window_start = now - 60.0
            while self._times and self._times[0] < window_start:
                self._times.popleft()
            if len(self._times) >= self.rpm:
                sleep_for = 60.0 - (now - self._times[0]) + 0.05
                if sleep_for > 0:
                    time.sleep(sleep_for)
                now = time.monotonic()
                window_start = now - 60.0
                while self._times and self._times[0] < window_start:
                    self._times.popleft()
            self._times.append(now)


class GeminiClient(LLMClient):
    def __init__(
        self,
        model: str = "gemini-2.0-flash",
        api_key: str | None = None,
        temperature: float = 0.0,
        max_output_tokens: int = 256,
        requests_per_minute: int = 12,
    ):
        # Imported lazily so the package can be parsed without google-genai installed.
        from google import genai

        key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError(
                "GOOGLE_API_KEY not set. Get a free key at "
                "https://aistudio.google.com/app/apikey and put it in .env."
            )

        self._genai = genai
        self._client = genai.Client(api_key=key)
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self._limiter = _RateLimiter(requests_per_minute)

    def generate(self, prompt: str, **kwargs) -> GenerationResult:
        self._limiter.acquire()
        return self._generate_with_retry(prompt, **kwargs)

    @retry(
        reraise=True,
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=1, max=20),
        retry=retry_if_exception_type(Exception),
    )
    def _generate_with_retry(self, prompt: str, **kwargs) -> GenerationResult:
        from google.genai import types as gt

        config = gt.GenerateContentConfig(
            temperature=kwargs.get("temperature", self.temperature),
            max_output_tokens=kwargs.get("max_output_tokens", self.max_output_tokens),
        )

        t0 = time.monotonic()
        resp = self._client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=config,
        )
        latency_ms = (time.monotonic() - t0) * 1000.0

        text = (resp.text or "").strip() if hasattr(resp, "text") else ""

        usage = LLMUsage()
        meta = getattr(resp, "usage_metadata", None)
        if meta is not None:
            usage.prompt_tokens = int(getattr(meta, "prompt_token_count", 0) or 0)
            usage.completion_tokens = int(getattr(meta, "candidates_token_count", 0) or 0)
            usage.total_tokens = int(getattr(meta, "total_token_count", 0) or 0)

        return GenerationResult(
            text=text,
            usage=usage,
            latency_ms=latency_ms,
            model=self.model,
        )
