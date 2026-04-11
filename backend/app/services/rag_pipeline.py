"""
RAG Pipeline Module
Orchestrates the complete Retrieval-Augmented Generation workflow.
Coordinates query translation, retrieval, and LLM generation.
"""

import uuid
from typing import Dict, Any, List
from app.core.config import client, MODEL_NAME_GENERATION
from app.services.query_translator import understand_query
from app.services.retriever import (
    dense_retrieval,
    sparse_retrieval,
    hybrid_fusion,
    rerank,
    indexer
)


# =====================================================
# CONTEXT BUILDING FROM RETRIEVED CLAUSES
# =====================================================

def build_context(clauses: List[Dict[str, Any]]) -> str:
    """
    Converts retrieved clauses into a formatted context string for LLM.
    
    Args:
        clauses: List of retrieved clause dictionaries
        
    Returns:
        Formatted context string ready for LLM input
    """
    if not clauses:
        return ""
    
    context_str = ""
    for i, c in enumerate(clauses):
        doc_id = c['metadata'].get('document_id', 'Unknown')
        context_str += f"Clause {i+1} [Document ID: {doc_id}]:\n{c['text']}\n\n"
    
    return context_str.strip()


# =====================================================
# GROUNDED ANSWER GENERATION WITH GUARDRAILS
# =====================================================

def generate_grounded_answer(
    query: str,
    context: str,
    intent: Dict[str, Any]
) -> str:
    """
    Generates an answer grounded strictly in retrieved context.
    Implements strict guardrails to prevent hallucination.
    
    Args:
        query: User's original query
        context: Retrieved and formatted clause context
        intent: Extracted query intent (agreement_type, clause_type)
        
    Returns:
        Grounded answer based on retrieved context, or rejection message
    """
    if not context:
        return "The information is not available in the retrieved clauses."
        
    # Build system prompt with strict grounding rules
    system_prompt = f"""You are a precise legal assistant. Your ONLY objective is to answer the user query grounded strictly in the provided clauses.

RULES (MUST BE ENFORCED EXPLICITLY):
1. Answer ONLY using the retrieved clauses below.
2. DO NOT use external knowledge. DO NOT assume or infer undocumented details.
3. If the answer is explicitly present -> provide it.
4. If partially present -> summarize ONLY available information.
5. If NOT present at all -> return EXACTLY: "The information is not available in the retrieved clauses."
6. ALWAYS include supporting clause quotes to back your answer.
7. DO NOT paraphrase the supporting quotes. Retain the exact legal phrasing.
8. Maintain focus primarily on '{intent.get('clause_type', 'Any')}' components if identified.

RETRIEVED CONTEXT:
{context}
"""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME_GENERATION,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            temperature=0.0,
            max_tokens=500
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Generation Error: {e}"


# =====================================================
# RAG PIPELINE ORCHESTRATOR
# =====================================================

class RAGPipeline:
    """
    Master RAG pipeline orchestrator.
    Coordinates all components in the retrieval-augmented generation workflow.
    """

    def __init__(self):
        """Initialize RAGPipeline with indexer reference."""
        self.indexer = indexer
        
    def ingest_documents(self, clauses: List[Dict[str, Any]]) -> None:
        """
        Ingests legal documents into the retrieval systems.
        
        Args:
            clauses: List of clause dictionaries to ingest
        """
        self.indexer.ingest_data(clauses)

    def run(self, user_query: str, k_initial: int = 15, top_k_final: int = 5) -> str:
        """
        Executes the complete RAG pipeline end-to-end.
        
        Pipeline stages:
        1. Query Understanding - Extract legal intent
        2. Parallel Retrieval - Dense semantic + sparse keyword search
        3. Hybrid Fusion - Combine and score results
        4. Reranking - Cross-encoder refinement
        5. Context Building - Format for LLM
        6. Answer Generation - Grounded LLM response
        
        Args:
            user_query: User's natural language query
            k_initial: Number of candidates in initial retrieval (default: 15)
            top_k_final: Number of final results to use for context (default: 5)
            
        Returns:
            Final grounded answer or fallback message
        """
        # Stage 1: Query Understanding - Extract intent
        intent = understand_query(user_query)
        
        # Stage 2 & 3: Parallel Search - Dense + Sparse Retrieval
        dense_res = dense_retrieval(user_query, filters=intent, top_k=k_initial)
        sparse_res = sparse_retrieval(user_query, top_k=k_initial)
        
        # Stage 4: Hybrid Fusion - Combine with weighted scoring
        fused_res = hybrid_fusion(dense_res, sparse_res)
        
        # Stage 5: Reranking - Cross-encoder refinement
        reranked_res = rerank(user_query, fused_res, top_k=top_k_final)
        
        # Safety Check: Reject if no high-quality results survive
        if not reranked_res:
            return "The information is not available in the retrieved clauses."
        
        # Stage 6a: Context Building - Format clauses for LLM
        context = build_context(reranked_res)
        
        # Stage 6b: Answer Generation - Grounded LLM response
        final_answer = generate_grounded_answer(user_query, context, intent)
        
        return final_answer


# =====================================================
# CONVENIENCE FUNCTION FOR SIMPLE USAGE
# =====================================================

def hybrid_rag_pipeline(user_query: str) -> str:
    """
    Convenience function to run RAG pipeline with default settings.
    
    Args:
        user_query: User's natural language query
        
    Returns:
        Final grounded answer
    """
    pipeline = RAGPipeline()
    return pipeline.run(user_query)


# =====================================================
# TEST HARNESS & INGESTION DEMO ROUTINE
# =====================================================

if __name__ == "__main__":
    # Example Standardized JSON Legal Structures
    sample_clauses = [
        {
            "id": str(uuid.uuid4()),
            "text": "The receiving party shall hold and maintain the Confidential Information in strictest confidence for the sole and exclusive benefit of the disclosing party.",
            "metadata": {"agreement_type": "NDA", "clause_type": "Confidentiality", "document_id": "doc_001"}
        },
        {
            "id": str(uuid.uuid4()),
            "text": "This Non-Disclosure Agreement shall terminate 2 years after the Effective Date, unless extended by mutual written agreement.",
            "metadata": {"agreement_type": "NDA", "clause_type": "Term and Termination", "document_id": "doc_001"}
        },
        {
            "id": str(uuid.uuid4()),
            "text": "In the event of termination without cause, the Employer shall pay the Employee 3 months of severance pay.",
            "metadata": {"agreement_type": "Employment", "clause_type": "Termination", "document_id": "doc_002"}
        }
    ]
    
    print(">>> Ingesting Legal Dataset Mock...")
    pipeline = RAGPipeline()
    pipeline.ingest_documents(sample_clauses)
    
    print("\n[Running Execution Tests]")
    test_queries = [
        "How long does the NDA term last?",
        "What happens if an employee is fired without cause?",
        "What is the penalty for bankruptcy during an acquisition?"
    ]
    
    for q in test_queries:
        print(f"\nQuery: '{q}'")
        print("-" * 50)
        answer = pipeline.run(q)
        print(f"Output:\n{answer}")
