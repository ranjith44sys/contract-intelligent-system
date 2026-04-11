"""
Retrieval Module
Handles data indexing, dense/sparse retrieval, hybrid fusion, and reranking.
"""

import numpy as np
from typing import List, Dict, Any, Optional
from app.core.config import (
    dense_model,
    cross_encoder,
    dense_collection
)
from rank_bm25 import BM25Okapi


# =====================================================
# CLAUSE INDEXER - DATA STORAGE & INDEXING
# =====================================================

class ClauseIndexer:
    """
    Manages indexing of legal clauses in both dense (vector) 
    and sparse (BM25) search systems.
    """

    def __init__(self):
        """Initialize empty corpus and BM25 index."""
        self.sparse_corpus: List[Dict[str, Any]] = []
        self.bm25: Optional[BM25Okapi] = None
        self.tokenized_corpus: List[List[str]] = []

    def ingest_data(self, clauses: List[Dict[str, Any]]) -> None:
        """
        Ingests legal clauses into both dense (ChromaDB) and sparse (BM25) indexes.
        
        Expected clause format:
        {
          "id": "unique_id",
          "text": "clause text",
          "metadata": {
            "agreement_type": "NDA",
            "clause_type": "Confidentiality",
            "document_id": "doc_001"
          }
        }
        
        Args:
            clauses: List of clause dictionaries
        """
        if not clauses:
            return

        ids = [c["id"] for c in clauses]
        texts = [c["text"] for c in clauses]
        metadatas = [c["metadata"] for c in clauses]
        
        # 1. Index in ChromaDB (Dense Vector Embeddings)
        dense_embeddings = dense_model.encode(
            texts,
            normalize_embeddings=True
        ).tolist()
        
        dense_collection.add(
            ids=ids,
            embeddings=dense_embeddings,
            documents=texts,
            metadatas=metadatas
        )

        # 2. Index with BM25 (Sparse Keyword Matching)
        self.sparse_corpus.extend(clauses)
        self.tokenized_corpus = [c["text"].lower().split() for c in self.sparse_corpus]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        
        print(f"Successfully ingested {len(clauses)} clauses.")


# Global indexer instance
indexer = ClauseIndexer()


# =====================================================
# DENSE RETRIEVAL (SEMANTIC SEARCH)
# =====================================================

def dense_retrieval(
    query: str,
    filters: Dict[str, Any],
    top_k: int = 10
) -> Dict[str, Any]:
    """
    Performs semantic search using dense vector embeddings (ChromaDB).
    
    Supports optional metadata filtering by agreement_type.
    
    Args:
        query: User query string
        filters: Dictionary with optional filtering criteria (e.g., agreement_type)
        top_k: Number of top results to retrieve
        
    Returns:
        Dictionary mapping document IDs to retrieval results with dense_score
    """
    # Encode query to vector embedding
    query_embedding = dense_model.encode(
        [query],
        normalize_embeddings=True
    ).tolist()
    
    # Build optional metadata filter
    where_clause = {}
    if filters and filters.get("agreement_type"):
        where_clause["agreement_type"] = filters["agreement_type"]

    try:
        # Query ChromaDB
        query_args = {
            "query_embeddings": query_embedding,
            "n_results": min(top_k, len(indexer.sparse_corpus)) if indexer.sparse_corpus else top_k
        }
        if where_clause:
            query_args["where"] = where_clause
            
        results = dense_collection.query(**query_args)
            
        # Parse results into structured format
        dense_results = {}
        if results['ids'] and len(results['ids'][0]) > 0:
            distances = results['distances'][0]
            ids = results['ids'][0]
            docs = results['documents'][0]
            metas = results['metadatas'][0]
            
            # Convert distance (L2 or Cosine) to similarity score
            for d, i, doc, meta in zip(distances, ids, docs, metas):
                sim_score = 1.0 / (1.0 + d)  # Transform distance to similarity
                dense_results[i] = {
                    "id": i,
                    "text": doc,
                    "metadata": meta,
                    "dense_score": sim_score
                }
        return dense_results
        
    except Exception as e:
        print(f"Dense retrieval error: {e}")
        return {}


# =====================================================
# SPARSE RETRIEVAL (KEYWORD SEARCH)
# =====================================================

def sparse_retrieval(query: str, top_k: int = 10) -> Dict[str, Any]:
    """
    Performs keyword-based search using BM25 ranking.
    
    Args:
        query: User query string
        top_k: Number of top results to retrieve
        
    Returns:
        Dictionary mapping document IDs to retrieval results with sparse_score
    """
    if not indexer.bm25 or not indexer.sparse_corpus:
        return {}
        
    # Tokenize query and compute BM25 scores
    tokenized_query = query.lower().split()
    scores = indexer.bm25.get_scores(tokenized_query)
    
    # Get top-k indices
    top_n_indices = np.argsort(scores)[::-1][:top_k]
    max_score = max(scores) if len(scores) > 0 and max(scores) > 0 else 1.0
    
    # Build results dictionary
    sparse_results = {}
    for idx in top_n_indices:
        score = scores[idx]
        if score <= 0:
            continue
            
        clause = indexer.sparse_corpus[idx]
        sparse_results[clause["id"]] = {
            "id": clause["id"],
            "text": clause["text"],
            "metadata": clause["metadata"],
            "sparse_score": score / max_score  # Normalize BM25 score
        }
    return sparse_results


# =====================================================
# HYBRID FUSION & DEDUPLICATION
# =====================================================

def hybrid_fusion(
    dense_res: Dict[str, Any],
    sparse_res: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Fuses dense and sparse retrieval results using weighted scoring.
    Combines semantic and keyword-based relevance signals.
    
    Weights: 70% dense score + 30% sparse score
    
    Args:
        dense_res: Results from dense retrieval
        sparse_res: Results from sparse retrieval
        
    Returns:
        List of fused results sorted by hybrid_score (descending)
    """
    combined = {}
    all_ids = set(dense_res.keys()).union(set(sparse_res.keys()))
    
    for doc_id in all_ids:
        d_item = dense_res.get(doc_id, {})
        s_item = sparse_res.get(doc_id, {})
        
        # Use whichever item has data
        base_item = d_item if d_item else s_item
        
        # Extract scores (default to 0 if not present)
        d_score = d_item.get("dense_score", 0.0)
        s_score = s_item.get("sparse_score", 0.0)
        
        # Weighted fusion: 70% dense + 30% sparse
        final_score = (0.7 * d_score) + (0.3 * s_score)
        
        combined[doc_id] = {
            "id": base_item["id"],
            "text": base_item["text"],
            "metadata": base_item["metadata"],
            "hybrid_score": final_score
        }
        
    # Sort by hybrid score (highest first)
    return sorted(combined.values(), key=lambda x: x["hybrid_score"], reverse=True)


# =====================================================
# RE-RANKING (CROSS-ENCODER)
# =====================================================

def rerank(
    query: str,
    candidates: List[Dict[str, Any]],
    top_k: int = 5,
    threshold: float = -10.0
) -> List[Dict[str, Any]]:
    """
    Re-ranks candidate documents using a cross-encoder model.
    Applies quality threshold to filter low-relevance results.
    
    Args:
        query: User query string
        candidates: List of candidate documents to rerank
        top_k: Number of top results to return after reranking
        threshold: Minimum relevance score threshold (docs below are filtered)
        
    Returns:
        List of top_k reranked results sorted by rerank_score (descending)
    """
    if not candidates:
        return []
        
    # Create query-document pairs for cross-encoder
    pairs = [[query, c["text"]] for c in candidates]
    
    # Get cross-encoder relevance scores
    scores = cross_encoder.predict(pairs)
    
    # Attach rerank scores to candidates
    for i, c in enumerate(candidates):
        c["rerank_score"] = float(scores[i])
        
    # Filter by confidence threshold
    filtered = [c for c in candidates if c["rerank_score"] > threshold]
    
    # Sort and return top_k
    sorted_reranked = sorted(
        filtered,
        key=lambda x: x["rerank_score"],
        reverse=True
    )
    
    return sorted_reranked[:top_k]
