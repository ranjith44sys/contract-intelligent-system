from typing import Dict, List
from app.config import client, MODEL
import json


class QueryTranslator:

    def _prompt(self, query: str) -> str:
        return f"""
You are an expert legal information retrieval assistant specializing in Retrieval-Augmented Generation (RAG) systems.

Your task is to transform a user query into structured, retrieval-optimized representations that maximize both recall and precision in downstream semantic and keyword search systems.

You must generate THREE outputs:

---

1. Multi-Query Expansion
- Generate 3 to 5 diverse query variants
- Each query should reflect a different perspective:
  • synonym variation (legal terminology)
  • paraphrasing
  • broader interpretation
  • narrower/specific interpretation
- Preserve legal intent and terminology accuracy
- Avoid redundancy — each query must be meaningfully distinct
- Ensure queries are retrieval-friendly (not too verbose)

---

2. Query Rewriting
- Convert the input into a clear, formal, well-structured natural language question
- Use precise legal phrasing where appropriate
- Make implicit intent explicit
- Avoid ambiguity

---

3. Query Decomposition (Least-to-Most Reasoning)
- Break the query into 2–4 smaller sub-queries
- Order them logically from foundational → advanced
- Each sub-query should:
  • be independently retrievable
  • focus on a single concept
  • collectively cover the full intent

---

Guidelines:
- Assume the domain is legal contracts unless specified otherwise
- Prefer legally meaningful terms (e.g., "indemnity", "breach", "liability", "termination")
- Handle vague or short queries by intelligently inferring intent
- Do NOT hallucinate facts — only reformulate the query
- Keep outputs concise and structured

---

Return ONLY valid JSON in the following format:

{
  "expanded": ["query1", "query2", "query3"],
  "rewritten": "clear rewritten question",
  "decomposed": ["step1", "step2", "step3"]
}

---

User Query:
"{query}"
"""

    def translate(self, query: str) -> Dict:

        response = client.responses.create(
            model=MODEL,
            temperature=0.2,
            response_format={"type": "json_object"},
            input=self._prompt(query)
        )

        parsed = json.loads(response.output[0].content[0].text)

        return {
            "original": query,
            "expanded": parsed["expanded"],
            "rewritten": parsed["rewritten"],
            "decomposed": parsed["decomposed"]
        }


'''
Examples: 

1. what happens if contract is breached

{
  "original": "what happens if contract is breached",
  "expanded": "contract OR agreement",
  "rewritten": "What are the clauses regarding what happens if contract is breached?",
  "decomposed": [
    "what clause",
    "happens clause",
    "contract clause"
  ]
}

2. fraud liability in contract

{
  "original": "fraud liability in contract",
  "expanded": "fraud OR misrepresentation OR deceit OR liability OR indemnity OR responsibility OR contract OR agreement",
  "rewritten": "What are the liability clauses related to fraud?",
  "decomposed": [
    "liability clause",
    "fraud condition"
  ]
}

3. audit rights of client

{
  "original": "audit rights of client",
  "expanded": "audit OR inspection OR review",
  "rewritten": "What are the audit rights and obligations defined in the contract?",
  "decomposed": [
    "audit clause",
    "rights clause",
    "client clause"
  ]
}

'''