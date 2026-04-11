import os
import json
import re
from app.schema.query import QueryTranslation
from app.utils.cache import get_cache, set_cache

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


class QueryTranslator:
    def __init__(self, api_key: str = None):
        # Allow passing API key, fallback to env var
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if OpenAI and self.api_key:
            self.client = OpenAI(api_key=self.api_key)
        else:
            self.client = None
        self.model = os.getenv("MODEL", "gpt-4-turbo")

    def translate(self, query: str) -> QueryTranslation:
        cache_key = f"query:{query}"
        cached = get_cache(cache_key)
        if cached:
            return QueryTranslation(**cached)

        if not self.client:
            # Fallback when no API Key is available
            expanded = [
                f"{query} legal definition",
                f"{query} liability clause",
                f"meaning of {query} in contract"
            ]
            data = {
                "expanded": expanded,
                "rewritten": f"What are the contract clauses regarding {query}?",
                "decomposed": [f"{query} rules", f"{query} conditions"]
            }
            set_cache(cache_key, data)
            return QueryTranslation(**data)

        # Build prompt and query
        user_message = f"Translate this legal contract query:\n\n{query}"

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a query translation engine for a legal contract semantic search RAG system.\n"
                            "Output ONLY valid JSON — no markdown, no commentary, no extra keys.\n\n"
                            "Rules:\n"
                            "- 'expanded': EXACTLY 4 paraphrased variants of the query. This field is MANDATORY "
                            "and must always contain 4 strings. Use legal synonyms, clause-level rephrasing, "
                            "jurisdiction-aware language, and different levels of specificity.\n"
                            "- 'rewritten': One precise legal query optimised for embedding search.\n"
                            "- 'decomposed': 2-3 atomic legal sub-questions ONLY for multi-issue queries. Use [] for single-issue.\n\n"
                            "Output format (strict):\n"
                            '{"expanded": ["legal v1", "legal v2", "legal v3", "legal v4"], '
                            '"rewritten": "precise legal query", "decomposed": ["sub-q1", "sub-q2"]}'
                        )
                    },
                    {"role": "user", "content": user_message}
                ],
                temperature=0.5,
            )
            content = response.choices[0].message.content.strip()
            content = re.sub(r"^```[a-z]*\n?|```$", "", content, flags=re.MULTILINE).strip()
            data = json.loads(content)
        except Exception:
            data = {
                "expanded": [query] * 4,
                "rewritten": query,
                "decomposed": [],
            }

        # --- Enforce and repair constraints ---
        expanded   = [s for s in data.get("expanded", []) if isinstance(s, str) and s.strip()]
        decomposed = [s for s in data.get("decomposed", []) if isinstance(s, str) and s.strip()]
        rewritten  = data.get("rewritten", query)

        # Deduplicate while preserving order
        seen = []
        for v in expanded:
            if v not in seen:
                seen.append(v)
        expanded = seen

        # Pad to minimum 3 with legal-aware fallbacks
        legal_fallbacks = [
            f"legal contract clause regarding: {query}",
            f"contractual obligation related to: {query}",
            f"agreement provision concerning: {query}",
        ]
        fb_idx = 0
        while len(expanded) < 3:
            expanded.append(legal_fallbacks[fb_idx % len(legal_fallbacks)])
            fb_idx += 1

        expanded = expanded[:5]
        decomposed = decomposed[:3]

        if not isinstance(rewritten, str) or not rewritten.strip():
            rewritten = query

        data = {
            "expanded": expanded,
            "rewritten": rewritten,
            "decomposed": decomposed,
        }

        set_cache(cache_key, data)
        return QueryTranslation(**data)