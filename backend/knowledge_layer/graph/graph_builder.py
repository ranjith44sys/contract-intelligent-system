from .graph_schema import NodeType, RelationType, create_node, create_edge
import logging

logger = logging.getLogger("knowledge_layer.graph")

class KnowledgeGraphBuilder:
    def __init__(self):
        self.nodes = []
        self.edges = []
        self.seen_nodes = set()

    def add_node(self, node):
        if node["node_id"] not in self.seen_nodes:
            self.nodes.append(node)
            self.seen_nodes.add(node["node_id"])
        return node["node_id"]

    def add_edge(self, source, target, relation):
        self.edges.append(create_edge(source, target, relation))

    def build_hierarchy(self, doc_id, doc_fields, clause_graph):
        """Constructs the hierarchical graph structure from document metadata and clause graph."""
        logger.info(f"Building knowledge graph for document: {doc_id}")
        
        # 1. Document Node
        doc_node_id = self.add_node(create_node(
            doc_id, 
            NodeType.DOCUMENT, 
            {"document_id": doc_id, **doc_fields}
        ))

        sections = {} # Map section_number -> section_node_id

        # 2. Process Clauses and Sections
        for clause in clause_graph:
            sec_num = clause.get("section_number", "UNSPECIFIED")
            
            # Create Section node if new
            if sec_num not in sections:
                sec_id = f"{doc_id}_SEC_{sec_num}"
                sections[sec_num] = self.add_node(create_node(
                    sec_id,
                    NodeType.SECTION,
                    {"section_number": sec_num, "document_id": doc_id}
                ))
                self.add_edge(doc_node_id, sections[sec_num], RelationType.HAS_SECTION)

            # Create Clause node
            clause_id = clause["clause_id"]
            clause_attr = {
                "clause_id": clause_id,
                "clause_title": clause.get("clause_title"),
                "clause_type": clause.get("clause_type"),
                "page_number": clause.get("page_number"),
                "text": clause["text"]
            }
            self.add_node(create_node(clause_id, NodeType.CLAUSE, clause_attr))
            self.add_edge(sections[sec_num], clause_id, RelationType.HAS_CLAUSE)

            # 2.5 Add Chunks linked to Clause (Detailed structure)
            for chunk in clause.get("chunks", []):
                chunk_id = chunk["chunk_id"]
                self.add_node(create_node(
                    chunk_id,
                    NodeType.CHUNK,
                    {"chunk_id": chunk_id, "text": chunk["text"], "page_number": chunk.get("page_number")}
                ))
                self.add_edge(clause_id, chunk_id, RelationType.HAS_CHUNK)

            # 3. Add Fields linked to Clause

            extracted_fields = clause.get("extracted_fields")
            if extracted_fields and isinstance(extracted_fields, dict):
                for field_name, field_value in extracted_fields.items():
                    if field_value:
                        field_id = f"{clause_id}_FLD_{field_name}"
                        self.add_node(create_node(
                            field_id,
                            NodeType.FIELD,
                            {"field_name": field_name, "value": field_value}
                        ))
                        self.add_edge(clause_id, field_id, RelationType.HAS_FIELD)

            # 4. Add Cross-references if any
            refs = clause.get("references", [])
            for ref_id in refs:
                # We assume ref_id might be another clause_id in this graph
                self.add_edge(clause_id, ref_id, RelationType.REFERENCES)

        return {
            "nodes": self.nodes,
            "edges": self.edges
        }
