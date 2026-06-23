"""All LLM prompt templates live here — one place to audit and tune.

Keeping prompts out of business logic makes them reviewable like any other
asset and lets us version/A-B them without touching pipeline code.
"""
from __future__ import annotations

# --------------------------------------------------------------------------
# Ingestion: combined enrichment (entities + topics + sentiment in ONE call)
# --------------------------------------------------------------------------
# We deliberately fuse the three extraction tasks into a single request.
# Rationale: 3x fewer Gemini calls (cost + latency), and the model reasons
# about entities/topics/sentiment with shared context, which improves
# consistency. The individual extractor classes remain available for targeted
# use, but the ingestion hot path uses this.
ENRICHMENT_PROMPT = """You are an information-extraction engine for a Reddit \
intelligence system. Analyse the CONTENT below and return STRICT JSON.

Return this exact schema:
{{
  "topics": ["3-6 concise lowercase topic phrases, e.g. 'rag systems', 'ai safety'"],
  "entities": [
    {{"name": "canonical name", "type": "MODEL|ORG|PERSON|PRODUCT|CONCEPT|DATASET"}}
  ],
  "sentiment": "positive|neutral|negative|mixed",
  "sentiment_score": -1.0 to 1.0
}}

Rules:
- Topics are themes, not entities. Normalise to lowercase noun phrases.
- Entities are specific named things (GPT-4, OpenAI, LangChain, Yann LeCun).
- Deduplicate. Prefer canonical spellings (e.g. "Llama 3" not "llama3").
- sentiment_score is the author's stance toward the main subject.
- Output ONLY JSON. No commentary.

SUBREDDIT: r/{subreddit}
CONTENT:
\"\"\"
{content}
\"\"\"
"""

ENTITY_ONLY_PROMPT = """Extract named entities from the text. Return STRICT JSON:
{{"entities": [{{"name": "...", "type": "MODEL|ORG|PERSON|PRODUCT|CONCEPT|DATASET"}}]}}
Only JSON. Text:
\"\"\"
{content}
\"\"\"
"""

TOPIC_ONLY_PROMPT = """Extract 3-6 lowercase topic phrases. Return STRICT JSON:
{{"topics": ["..."]}}
Only JSON. Text:
\"\"\"
{content}
\"\"\"
"""

SENTIMENT_ONLY_PROMPT = """Classify the author's sentiment toward the main \
subject. Return STRICT JSON:
{{"sentiment": "positive|neutral|negative|mixed", "sentiment_score": -1.0 to 1.0}}
Only JSON. Text:
\"\"\"
{content}
\"\"\"
"""

# --------------------------------------------------------------------------
# Retrieval: query understanding / routing
# --------------------------------------------------------------------------
QUERY_UNDERSTANDING_PROMPT = """You route queries for a hybrid GraphRAG system \
over Reddit data. Two retrievers exist:
  - GRAPH: best for relationships, influence, "who/which community", \
aggregations, counting, traversals.
  - VECTOR: best for semantic / "what are people saying about X" / fuzzy \
topical questions.

Analyse the QUERY and return STRICT JSON:
{{
  "normalized_query": "a clean search-optimised restatement",
  "intent": "semantic|relational|hybrid|temporal",
  "topics": ["topic phrases to match in the graph"],
  "entities": ["named entities of interest"],
  "subreddits": ["subreddits if explicitly mentioned, else empty"],
  "graph_weight": 0.0-2.0,
  "vector_weight": 0.0-2.0
}}

Guidance:
- "who/which/most influential/leading communities" -> relational, graph_weight high.
- "what are people saying / opinions / explain" -> semantic, vector_weight high.
- mixed -> hybrid, both ~1.0.
- Do NOT parse dates; another component handles time.
- Output ONLY JSON.

QUERY: {query}
"""

# --------------------------------------------------------------------------
# Answer generation
# --------------------------------------------------------------------------
ANSWER_PROMPT = """You are an analyst for a Reddit consumer-intelligence \
platform. Answer the QUESTION using ONLY the numbered SOURCES below. Each \
source is a real Reddit post or comment with a timestamp and subreddit.

Requirements:
- Cite every claim with inline markers like [1], [2] referring to source numbers.
- If sources span multiple time periods, describe how the discussion evolved.
- If the sources are insufficient, say so plainly — do not invent facts.
- Be concise, specific, and analytical. Quote sparingly.

QUESTION: {question}
{temporal_note}
SOURCES:
{sources}

Answer:"""

COMPARISON_NOTE = """This is a time-comparison question. The sources are \
grouped by period. Explicitly contrast the periods (what emerged, what faded, \
what shifted in sentiment or volume).
"""
