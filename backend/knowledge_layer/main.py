import os
import json
import logging
from .embeddings.embedder import embedder
from .embeddings.chroma_store import chroma_store
from .graph.graph_builder import KnowledgeGraphBuilder

logger = logging.getLogger("knowledge_layer")

class KnowledgeLayer:
    def __init__(self, output_dir=None):
        self.output_dir = output_dir or os.path.join(os.path.dirname(__file__), "output")
        os.makedirs(self.output_dir, exist_ok=True)

    def process_results(self, final_graph, doc_fields, document_name="Unknown", doc_id=None):
        """Processes the final pipeline output into embeddings and knowledge graph."""
        logger.info("Knowledge Layer processing started...")
        
        # Use provided doc_id (job_id) or fallback to doc_fields
        if not doc_id:
            doc_id = doc_fields.get("document_id", "DOC_UNKNOWN")
            if doc_id == "DOC_UNKNOWN" and final_graph:
                doc_id = final_graph[0].get("contract_id", "DOC_UNKNOWN")

        # 1. Generate Embeddings and Save to ChromaDB
        self._process_embeddings(doc_id, final_graph, document_name)
        
        # 2. Build and Save Knowledge Graph
        graph_data = self._process_graph(doc_id, doc_fields, final_graph)
        
        return {
            "chroma_db_path": os.path.join(self.output_dir, "chroma_db"),
            "graph_file": os.path.join(self.output_dir, "knowledge_graph.json"),
            "node_count": len(graph_data["nodes"]),
            "edge_count": len(graph_data["edges"])
        }

    def _process_embeddings(self, doc_id, final_graph, document_name):
        """Vectorizes clauses and chunks, then stores them in ChromaDB meticulously with filename tags."""
        ids = []
        embeddings = []
        documents = []
        metadatas = []

        for clause in final_graph:
            # Embed the full clause
            clause_id = clause["clause_id"]
            ids.append(clause_id)
            embeddings.append(embedder.embed_text(clause["text"]))
            documents.append(clause["text"])
            metadatas.append({
                "type": "clause",
                "document_id": doc_id,
                "document_name": document_name,
                "clause_id": clause_id,
                "clause_type": clause.get("clause_type"),
                "page_number": clause.get("page_number"),
                "section_number": clause.get("section_number")
            })

            # Embed each chunk
            for chunk in clause.get("chunks", []):
                chunk_id = chunk["chunk_id"]
                ids.append(chunk_id)
                embeddings.append(embedder.embed_text(chunk["text"]))
                documents.append(chunk["text"])
                metadatas.append({
                    "type": "chunk",
                    "document_id": doc_id,
                    "document_name": document_name,
                    "clause_id": clause_id,
                    "chunk_id": chunk_id,
                    "page_number": clause.get("page_number")
                })

        # Filter out None embeddings (empty text)
        valid_indices = [i for i, emb in enumerate(embeddings) if emb is not None]
        ids = [ids[i] for i in valid_indices]
        embeddings = [embeddings[i] for i in valid_indices]
        documents = [documents[i] for i in valid_indices]
        metadatas = [metadatas[i] for i in valid_indices]

        if ids:
            chroma_store.add_documents(ids, embeddings, documents, metadatas)

    def _process_graph(self, doc_id, doc_fields, final_graph):
        """Constructs and saves/merges the hierarchical knowledge graph with high resilience."""
        try:
            builder = KnowledgeGraphBuilder()
            new_graph_data = builder.build_hierarchy(doc_id, doc_fields, final_graph)
            
            output_file = os.path.join(self.output_dir, "knowledge_graph.json")
            
            # 1. Load existing graph if available
            global_nodes = {}
            global_edges = []
            
            if os.path.exists(output_file):
                try:
                    with open(output_file, "r") as f:
                        old_data = json.load(f)
                        if isinstance(old_data, dict):
                            global_nodes = {n["node_id"]: n for n in old_data.get("nodes", []) if isinstance(n, dict) and "node_id" in n}
                            global_edges = [e for e in old_data.get("edges", []) if isinstance(e, dict)]
                except (json.JSONDecodeError, Exception) as e:
                    logger.warning(f"Failed to load existing knowledge graph: {e}. Starting fresh.")

            # 2. Merge new nodes (uniquify by node_id)
            for node in new_graph_data.get("nodes", []):
                if isinstance(node, dict) and "node_id" in node:
                    global_nodes[node["node_id"]] = node
                
            # 3. Merge new edges (avoid exact duplicates)
            new_edge_ids = set()
            for e in global_edges:
                edge_sig = (e.get("from"), e.get("to"), e.get("type"))
                if all(edge_sig):
                    new_edge_ids.add(edge_sig)
                
            for edge in new_graph_data.get("edges", []):
                if not isinstance(edge, dict): continue
                edge_tuple = (edge.get("from"), edge.get("to"), edge.get("type"))
                if all(edge_tuple) and edge_tuple not in new_edge_ids:
                    global_edges.append(edge)
                    new_edge_ids.add(edge_tuple)

            # 4. Save merged graph ATOMICALLY
            merged_data = {
                "nodes": list(global_nodes.values()),
                "edges": global_edges
            }
            
            temp_file = output_file + ".tmp"
            with open(temp_file, "w") as f:
                json.dump(merged_data, f, indent=2)
            
            # Atomic swap to avoid file locks stalling the process
            os.replace(temp_file, output_file)
                
            logger.info(f"Knowledge Graph successfully merged and saved to {output_file}")
            return merged_data
            
        except Exception as e:
            logger.error(f"CRITICAL: Knowledge Graph merge failed for {doc_id}: {e}")
            # Return fresh data at minimum so the pipeline can continue
            return new_graph_data if 'new_graph_data' in locals() else {"nodes": [], "edges": []}
