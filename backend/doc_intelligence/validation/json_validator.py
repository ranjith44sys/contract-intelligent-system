import jsonschema
from ..config import logger

CLAUSE_GRAPH_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "clause_id": {"type": "string"},
            "contract_id": {"type": "string"},
            "contract_type": {"type": "string"},
            "clause_title": {"type": "string"},
            "clause_type": {"type": "string"},
            "confidence": {"type": "number"},
            "text": {"type": "string"},
            "raw_text": {"type": "string"},
            "page_number": {"type": "integer"},
            "section_number": {"type": "string"},
            "definitions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "term": {"type": "string"},
                        "definition": {"type": "string"}
                    }
                }
            },
            "references": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "type": {"type": "string"},
                        "value": {"type": "string"}
                    }
                }
            },
            "chunks": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "chunk_id": {"type": "string"},
                        "text": {"type": "string"},
                        "token_count": {"type": "integer"}
                    }
                }
            }
        },
        "required": ["clause_id", "contract_type", "clause_title", "clause_type", "text"]
    }
}

class JSONValidator:
    @staticmethod
    def validate_graph(graph_data):
        """Validates the clause graph against the JSON schema."""
        try:
            jsonschema.validate(instance=graph_data, schema=CLAUSE_GRAPH_SCHEMA)
            logger.info("Clause graph JSON validation successful.")
            return True, None
        except jsonschema.exceptions.ValidationError as e:
            logger.error(f"JSON Validation Error: {e.message}")
            return False, e.message
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return False, str(e)
