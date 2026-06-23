"""Thin, resilient wrapper around Gemini 2.5 Flash.

Responsibilities:
    * Lazy client init (so importing this module never requires a key).
    * JSON-mode generation with robust parsing (handles ```json fences).
    * Exponential-backoff retries on transient errors (tenacity).
    * Graceful degradation: if no API key is configured and fallback is
      allowed, callers receive `None` and switch to heuristics instead of
      crashing — which keeps the demo runnable offline.

Gemini 2.5 Flash is the right model here: extraction + routing + synthesis
are high-volume, latency-sensitive, and not reasoning-heavy. Flash gives ~10x
cheaper tokens than Pro at easily sufficient quality for structured
extraction, and its native JSON mode removes brittle output parsing.
"""
from __future__ import annotations

import json
from typing import Any, Optional

from tenacity import retry, stop_after_attempt, wait_exponential

from config import get_settings
from utils.logging import get_logger

logger = get_logger("llm.gemini")


class GeminiClient:
    def __init__(self) -> None:
        self._settings = get_settings()
        self._model = None  # lazy

    # ---- lifecycle ----------------------------------------------------
    @property
    def enabled(self) -> bool:
        return self._settings.llm_enabled

    def _ensure_model(self):
        if self._model is not None:
            return self._model
        if not self.enabled:
            raise RuntimeError("GEMINI_API_KEY not configured")
        import google.generativeai as genai  # imported lazily

        genai.configure(api_key=self._settings.gemini_api_key)
        self._model = genai.GenerativeModel(self._settings.gemini_model)
        logger.info("Gemini model initialised: %s", self._settings.gemini_model)
        return self._model

    # ---- generation ---------------------------------------------------
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=20),
        reraise=True,
    )
    def _generate(self, prompt: str, *, json_mode: bool, temperature: float) -> str:
        import google.generativeai as genai

        model = self._ensure_model()
        cfg = genai.types.GenerationConfig(
            temperature=temperature,
            response_mime_type="application/json" if json_mode else "text/plain",
        )
        resp = model.generate_content(prompt, generation_config=cfg)
        return resp.text or ""

    def generate_text(
        self, prompt: str, *, temperature: float = 0.3
    ) -> Optional[str]:
        if not self.enabled:
            return None
        try:
            return self._generate(prompt, json_mode=False, temperature=temperature)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Gemini text generation failed: %s", exc)
            if not self._settings.llm_allow_fallback:
                raise
            return None

    def generate_json(
        self, prompt: str, *, temperature: float = 0.0, default: Any = None
    ) -> Any:
        """Return parsed JSON, or `default` on any failure (when fallback ok)."""
        if not self.enabled:
            return default
        try:
            raw = self._generate(prompt, json_mode=True, temperature=temperature)
            return self._parse_json(raw, default)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Gemini JSON generation failed: %s", exc)
            if not self._settings.llm_allow_fallback:
                raise
            return default

    @staticmethod
    def _parse_json(raw: str, default: Any) -> Any:
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.strip("`")
            raw = raw[4:] if raw.lower().startswith("json") else raw
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            start = raw.find("{")
            start = raw.find("[") if start == -1 else start
            end = max(raw.rfind("}"), raw.rfind("]"))
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(raw[start : end + 1])
                except json.JSONDecodeError:
                    pass
            logger.warning("Could not parse Gemini JSON; returning default")
            return default


_client: Optional[GeminiClient] = None


def get_gemini_client() -> GeminiClient:
    global _client
    if _client is None:
        _client = GeminiClient()
    return _client
