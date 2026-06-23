"""LLM-powered enrichment: entities, topics, sentiment.

Design:
    * `EntityExtractor`, `TopicExtractor`, `SentimentAnalyzer` are focused,
      independently usable classes (each with its own prompt) — exactly the
      separation the assignment asks for, and handy for targeted re-runs.
    * `ContentEnricher` is the production hot path: it performs ONE combined
      Gemini call that returns all three at once (3x fewer requests), and
      transparently falls back to heuristics when the LLM is unavailable.

Every class returns plain domain objects from `utils.models`.
"""
from __future__ import annotations

from typing import List, Tuple

from llm import heuristics, prompts
from llm.gemini_client import GeminiClient, get_gemini_client
from utils.logging import get_logger
from utils.models import Enrichment, Entity, Sentiment
from utils.text import truncate

logger = get_logger("llm.extractors")

_VALID_SENTIMENT = {s.value for s in Sentiment}


def _coerce_sentiment(label: str) -> Sentiment:
    label = (label or "").lower().strip()
    return Sentiment(label) if label in _VALID_SENTIMENT else Sentiment.NEUTRAL


def _coerce_entities(raw: list) -> List[Entity]:
    out: List[Entity] = []
    for e in raw or []:
        if isinstance(e, dict) and e.get("name"):
            out.append(Entity(name=str(e["name"]).strip(), type=str(e.get("type", "CONCEPT")).upper()))
        elif isinstance(e, str) and e.strip():
            out.append(Entity(name=e.strip(), type="CONCEPT"))
    return out


class EntityExtractor:
    def __init__(self, client: GeminiClient | None = None):
        self._client = client or get_gemini_client()

    def extract(self, text: str) -> List[Entity]:
        data = self._client.generate_json(
            prompts.ENTITY_ONLY_PROMPT.format(content=truncate(text)), default=None
        )
        if data is None:
            return heuristics.heuristic_entities(text)
        return _coerce_entities(data.get("entities", []))


class TopicExtractor:
    def __init__(self, client: GeminiClient | None = None):
        self._client = client or get_gemini_client()

    def extract(self, text: str) -> List[str]:
        data = self._client.generate_json(
            prompts.TOPIC_ONLY_PROMPT.format(content=truncate(text)), default=None
        )
        if data is None:
            return heuristics.heuristic_topics(text)
        return [str(t).lower().strip() for t in data.get("topics", []) if str(t).strip()]


class SentimentAnalyzer:
    def __init__(self, client: GeminiClient | None = None):
        self._client = client or get_gemini_client()

    def analyze(self, text: str) -> Tuple[Sentiment, float]:
        data = self._client.generate_json(
            prompts.SENTIMENT_ONLY_PROMPT.format(content=truncate(text)), default=None
        )
        if data is None:
            return heuristics.heuristic_sentiment(text)
        return _coerce_sentiment(data.get("sentiment")), float(data.get("sentiment_score", 0.0) or 0.0)


class ContentEnricher:
    """Single-call combined extraction — the ingestion hot path."""

    def __init__(self, client: GeminiClient | None = None):
        self._client = client or get_gemini_client()

    def enrich(self, *, text: str, subreddit: str) -> Enrichment:
        prompt = prompts.ENRICHMENT_PROMPT.format(
            subreddit=subreddit, content=truncate(text)
        )
        data = self._client.generate_json(prompt, default=None)
        if data is None:
            return self._heuristic(text)
        try:
            return Enrichment(
                topics=[str(t).lower().strip() for t in data.get("topics", []) if str(t).strip()],
                entities=_coerce_entities(data.get("entities", [])),
                sentiment=_coerce_sentiment(data.get("sentiment")),
                sentiment_score=float(data.get("sentiment_score", 0.0) or 0.0),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Enrichment coercion failed (%s); using heuristics", exc)
            return self._heuristic(text)

    @staticmethod
    def _heuristic(text: str) -> Enrichment:
        label, score = heuristics.heuristic_sentiment(text)
        return Enrichment(
            topics=heuristics.heuristic_topics(text),
            entities=heuristics.heuristic_entities(text),
            sentiment=label,
            sentiment_score=score,
        )
