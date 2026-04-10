import uuid
from ..config import logger

class ClauseGraphBuilder:
    def __init__(self, contract_type, contract_id=None):
        self.contract_type = contract_type
        self.contract_id = contract_id or f"C_{uuid.uuid4().hex[:8].upper()}"

    def build_graph(self, segmented_clauses):
        """Builds the initial graph from segmented clauses."""
        graph = []
        
        for i, clause in enumerate(segmented_clauses):
            clause_id = f"{self.contract_id}_CL_{i+1:03d}"
            
            node = {
                "clause_id": clause_id,
                "contract_id": self.contract_id,
                "contract_type": self.contract_type,
                "clause_title": clause.get("clause_title", "Untitled Clause"),
                "clause_type": clause.get("clause_type", "Others"),
                "confidence": clause.get("confidence", 0.0),
                "text": clause.get("text", ""),
                "raw_text": clause.get("text", ""), # For now
                "page_number": clause.get("page_number", 1),
                "section_number": clause.get("section_number", "")
            }
            graph.append(node)
            
        return graph
