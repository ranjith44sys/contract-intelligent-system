from enum import Enum

class NodeType(str, Enum):
    DOCUMENT = "Document"
    SECTION = "Section"
    CLAUSE = "Clause"
    CHUNK = "Chunk"
    FIELD = "Field"
    ENTITY = "Entity"

class RelationType(str, Enum):
    HAS_SECTION = "HAS_SECTION"
    HAS_CLAUSE = "HAS_CLAUSE"
    HAS_CHUNK = "HAS_CHUNK"
    HAS_FIELD = "HAS_FIELD"
    REFERENCES = "REFERENCES"
    CONTAINS_ENTITY = "CONTAINS_ENTITY"

def create_node(node_id, node_type: NodeType, attributes: dict):
    """Factory function for creating consistent graph nodes."""
    return {
        "node_id": node_id,
        "type": node_type.value,
        "attributes": attributes
    }

def create_edge(source_id, target_id, relation: RelationType):
    """Factory function for creating consistent graph edges."""
    return {
        "source": source_id,
        "target": target_id,
        "relation": relation.value
    }
