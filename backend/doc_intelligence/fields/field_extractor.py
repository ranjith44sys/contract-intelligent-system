import json
import re
from ..utils import client
from ..config import logger

class FieldExtractor:
    def __init__(self):
        self.doc_level_taxonomy = [
            "Parties involved (names, roles)",
            "Effective date",
            "Expiration/Termination date",
            "Governing law",
            "Payment terms (if any)"
        ]
        
        self.clause_level_taxonomy = [
            "Payment Terms",
            "Liability Limit",
            "Notice Period",
            "Penalties",
            "Termination Conditions"
        ]

    def _normalize_data(self, data):
        """Normalizes extracted data, ensuring 'Not Found' defaults."""
        for key, value in data.items():
            if value is None or (isinstance(value, str) and not value.strip()):
                data[key] = "Not Found"
            
            # Simple ISO Date normalization Attempt (YYYY-MM-DD)
            if "date" in key.lower() and isinstance(value, str) and value != "Not Found":
                date_match = re.search(r'(\d{1,2})[/-](\d{1,2})[/-](\d{4})', value)
                if date_match:
                    data[key] = f"{date_match.group(3)}-{date_match.group(1).zfill(2)}-{date_match.group(2).zfill(2)}"

        return data

    def extract_document_fields(self, full_text):
        """Extracts contract-level fields with strict rules."""
        logger.info("Extracting document-level fields with strict rules...")
        
        system_prompt = (
            "You are a professional legal information extraction system. "
            "Analyze the contract text, which may contain OCR noise or missing segments. "
            "1. Extract the requested fields. "
            "2. If a field is not explicitly available, return 'Not Found'. "
            "3. DO NOT return empty strings or null. "
            "4. Interpret noisy or incomplete text intelligently to infer the correct values. "
            "Return ONLY a clean JSON object."
        )
        
        user_prompt = f"""
Extract these fields:
{chr(10).join(['- ' + f for f in self.doc_level_taxonomy])}

Document Text (truncated if necessary):
{full_text[:15000]} 

Expected JSON Format:
{{
  "parties": [ {{"name": "...", "role": "..."}} ],
  "effective_date": "YYYY-MM-DD or 'Not Found'",
  "termination_date": "YYYY-MM-DD or 'Not Found'",
  "governing_law": "...",
  "payment_terms": "..."
}}
"""
        try:
            response = client.get_completion(system_prompt, user_prompt, json_mode=True)
            if not response:
                return {k: "Not Found" for k in ["parties", "effective_date", "termination_date", "governing_law", "payment_terms"]}
            
            data = json.loads(response)
            return self._normalize_data(data)
        except Exception as e:
            logger.error(f"Document-level field extraction failed: {e}")
            return {}

    def extract_clause_fields(self, clause_text):
        """Extracts clause-specific attributes with strict rules."""
        system_prompt = (
            "You are a specialized legal analyst. Analyze the clause text (may have OCR noise). "
            "1. Extract the attributes below. "
            "2. Use 'Not Found' if an attribute is missing. "
            "3. Interpret noise intelligently. "
            "Return ONLY valid JSON."
        )
        
        user_prompt = f"""
Extract these fields if relevant to the clause:
{chr(10).join(['- ' + f for f in self.clause_level_taxonomy])}

Clause Text:
{clause_text}

Expected JSON Format:
{{
  "payment_terms": "...",
  "liability_limit": "...",
  "notice_period": "...",
  "penalty_terms": "...",
  "termination_conditions": "..."
}}
"""

        try:
            response = client.get_completion(system_prompt, user_prompt, json_mode=True)
            if not response:
                return {}
            
            data = json.loads(response)
            return data
        except Exception as e:
            logger.error(f"Clause-level field extraction failed: {e}")
            return {}

    def process_graph(self, graph, full_text):
        """Runs the complete field extraction workflow on the clause graph."""
        logger.info("Starting comprehensive field extraction...")
        
        # 1. Document Level
        doc_fields = self.extract_document_fields(full_text)
        
        # 2. Clause Level
        clause_level_results = []
        for clause in graph:
            # We only extract fields for relevant clause types to save costs
            relevant_types = ["Payment", "Liability", "Termination", "Notice", "Confidentiality", "Indemnification"]
            is_relevant = any(t.lower() in clause.get("clause_type", "").lower() for t in relevant_types)
            
            if is_relevant:
                logger.info(f"Extracting fields for clause: {clause['clause_id']}")
                fields = self.extract_clause_fields(clause["text"])
                clause["extracted_fields"] = fields
                
                clause_level_results.append({
                    "clause_id": clause["clause_id"],
                    "extracted_fields": fields,
                    "section_number": clause.get("section_number"),
                    "page_number": clause.get("page_number")
                })
            else:
                clause["extracted_fields"] = None

        return {
            "document_fields": doc_fields,
            "clause_level_fields": clause_level_results
        }
