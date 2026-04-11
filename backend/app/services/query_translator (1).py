"""
Query Translation Module
Handles query understanding and multi-faceted query transformation for RAG.
"""

import json
from typing import Dict, Any, Optional
from app.core.config import client, MODEL_NAME

# =====================================================
# QUERY UNDERSTANDING
# =====================================================

def understand_query(query: str) -> Dict[str, Any]:
    """
    Analyzes a user query to extract legal intent components.
    
    Extracts:
    - agreement_type: Type of legal agreement (e.g., NDA, Employment, Service Agreement)
    - clause_type: Specific clause type being queried (e.g., Confidentiality, Termination)
    
    Args:
        query: User's natural language query
        
    Returns:
        Dictionary with agreement_type and clause_type (or null if not found)
    """
    prompt = f"""
    You are a legal intent extraction AI component. 
    Analyze the following user query and extract the 'agreement_type' and 'clause_type'.
    If either is not found, output null for that field. Do NOT hallucinate types.
    
    Output strictly in JSON format:
    {{
      "agreement_type": "<type> or null",
      "clause_type": "<type> or null"
    }}
    
    User Query: "{query}"
    """
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.0,
            max_tokens=300
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Query understanding error: {e}")
        return {"agreement_type": None, "clause_type": None}


# =====================================================
# QUERY TRANSLATOR CLASS
# =====================================================

class QueryTranslator:
    """
    Transforms user queries into structured, retrieval-optimized representations.
    Implements multi-faceted query expansion for improved RAG performance.
    """

    def __init__(self):
        """Initialize QueryTranslator with LLM client."""
        self.client = client
        self.model = MODEL_NAME

    def _build_prompt(self, query: str) -> str:
        """
        Constructs the prompt for query transformation.
        
        Args:
            query: User's natural language query
            
        Returns:
            Formatted prompt for LLM
        """
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

{{
  "expanded": ["query1", "query2", "query3"],
  "rewritten": "clear rewritten question",
  "decomposed": ["step1", "step2", "step3"]
}}

---

User Query:
"{query}"
"""

    def translate(self, query: str) -> Dict[str, Any]:
        """
        Translates a user query into expanded, rewritten, and decomposed forms.
        
        Args:
            query: User's natural language query
            
        Returns:
            Dictionary containing original query and its transformations
        """
        prompt = self._build_prompt(query)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.2,
                max_tokens=800
            )
            
            parsed = json.loads(response.choices[0].message.content)
            
            return {
                "original": query,
                "expanded": parsed.get("expanded", []),
                "rewritten": parsed.get("rewritten", query),
                "decomposed": parsed.get("decomposed", [])
            }
        except Exception as e:
            print(f"Query translation error: {e}")
            return {
                "original": query,
                "expanded": [query],
                "rewritten": query,
                "decomposed": [query]
            }

