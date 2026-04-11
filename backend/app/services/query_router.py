import os
import json
from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage

# ==========================================
# 1. Define the Structured Output Schema for Query Analysis
# ==========================================
class QueryDetails(BaseModel):
    original_query: str = Field(..., description="The exact original sub-query extracted without modification.")
    quality_score: float = Field(..., ge=0.0, le=1.0, description="Score based on query clarity.")
    needs_rewrite: bool = Field(..., description="True if score < 0.60, False otherwise.")
    final_query: str = Field(..., description="The rewritten query, or exact original query if needs_rewrite is False.")
    document_type: Optional[str] = Field(None, description="Extracted classification of the agreement/document type (e.g., 'lease agreement', 'service agreement'). Null if unidentifiable.")
    clause_type: Optional[str] = Field(None, description="Extracted classification of the clause type (e.g., 'payment terms', 'termination clause', 'liability'). Null if unidentifiable.")

class MultiQueryEvaluationResult(BaseModel):
    is_multi_query: bool = Field(..., description="True if the user query contains multiple independent questions/intents.")
    queries: List[QueryDetails] = Field(..., description="The processed sub-queries and their extracted metadata.")

# ==========================================
# 2. System Prompt Definition for Analyzer
# ==========================================
ROUTER_SYSTEM_PROMPT = """You are the first phase of a legal Hybrid RAG system: The Query Analyzer & Router.

Your job is to process a raw user query, handle multiple sub-queries if present, evaluate their quality, selectively rewrite them, and EXTRACT key metadata parameters (document type and clause type) for the upcoming Vector Database search.

### STEP 1: QUERY ANALYSIS & SEGMENTATION
* If multiple intents are requested, split them into independent sub-queries.
* DO NOT merge queries. Preserve original intent exactly.

### STEP 2: QUALITY EVALUATION & REWRITING
* Quality Score: 0.85-1.0 (Clear), 0.6-0.84 (Minor issues), 0.0-0.59 (Unclear, requires rewrite).
* Rewrite ONLY if unclear, ambiguous, or unsuitable for semantic search.
* If rewriting, preserve EXACT meaning. Output the rewritten string in `final_query`.

### STEP 3: METADATA EXTRACTION FOR VECTOR DB
For each sub-query, you must extract:
1. `document_type`: The category of the agreement governing the query (e.g., "lease agreement", "service agreement"). 
2. `clause_type`: The topic or specific clause the user wants (e.g., "payment terms", "termination").
* If the user doesn't specify one of these, you may leave it Null, but extract it if it's explicitly stated.

* DO NOT ANSWER THE QUESTIONS. Your only job is to return the strict JSON schema formatting these search parameters.
"""

# ==========================================
# 3. Dummy Vector DB Retriever (To simulate querying your DB)
# ==========================================
def fetch_from_vector_db(final_query: str, doc_type: str, clause_type: str) -> str:
    """
    In production, this connects to MongoDB / Pinecone / ChromaDB, 
    pre-filters by metadata (doc_type, clause_type), and performs hybrid dense+sparse retrieval.
    """
    print(f"   [VectorDB Searching] Filtering by doc_type='{doc_type}' AND clause_type='{clause_type}'...")
    
    # Mocking what the database would return based on the inputs
    if clause_type == "payment terms":
        return "Extract from Lease Agreement DB: Tenant shall pay landlord $2000 per month due on the 1st. Late payments incur a 5% penalty."
    elif clause_type and "termination" in clause_type.lower():
        return "Extract from Service Agreement DB: Either party may terminate with 30 days written notice. Early termination requires payout of remaining balance."
    elif clause_type and "liability" in clause_type.lower():
        return "Extract from Contract DB: Total liability is capped at the amount paid under this agreement over the trailing 12 months."
    
    return "No relevant clauses found in the database."

# ==========================================
# 4. Agent Class Implementation
# ==========================================
class LegalRAGPipeline:
    def __init__(self, model_name: str = "openai/gpt-4o", temperature: float = 0.0):
        self.api_key = os.environ.get("OPENAI_API_KEY")
        self.base_url = "https://openrouter.ai/api/v1" if self.api_key and self.api_key.startswith("sk-or") else None
        
        # Analyzer Agent (Structured Output)
        self.analyzer_llm = ChatOpenAI(
            model=model_name, 
            temperature=temperature, 
            max_tokens=1000,
            api_key=self.api_key, 
            base_url=self.base_url,
            default_headers={"HTTP-Referer": "http://localhost", "X-Title": "Local RAG"} if self.base_url else None
        )
        self.structured_analyzer = self.analyzer_llm.with_structured_output(MultiQueryEvaluationResult)
        self.analyzer_prompt = ChatPromptTemplate.from_messages([
            ("system", ROUTER_SYSTEM_PROMPT),
            ("human", "User Query:\n{user_query}")
        ])
        
        self.analyzer_chain = self.analyzer_prompt | self.structured_analyzer

        # QA Generation Agent (Standard String Output)
        self.qa_llm = ChatOpenAI(
            model=model_name, 
            temperature=0.0,
            max_tokens=1000,
            api_key=self.api_key, 
            base_url=self.base_url,
            default_headers={"HTTP-Referer": "http://localhost", "X-Title": "Local RAG QA"} if self.base_url else None
        )

    def execute_pipeline(self, user_query: str) -> dict:
        print("\n=== PHASE 1: QUERY ANALYSIS & METADATA EXTRACTION ===")
        analysis_result: MultiQueryEvaluationResult = self.analyzer_chain.invoke({"user_query": user_query})
        
        final_responses = []
        
        print(f"\n=== PHASE 2: VECTOR DB RETRIEVAL AND RAG GENERATION ===")
        for i, query_data in enumerate(analysis_result.queries):
            print(f"\n-----------------------------------------------------")
            print(f"--- Sub-Query {i+1} Metrics ---")
            print(f"   Original Query : '{query_data.original_query}'")
            print(f"   Quality Score  : {query_data.quality_score} / 1.0")
            print(f"   Needs Rewrite  : {query_data.needs_rewrite}")
            print(f"   Final Query    : '{query_data.final_query}'")
            print(f"   Extracted Doc  : {query_data.document_type}")
            print(f"   Extracted Clause: {query_data.clause_type}")
            print(f"-----------------------------------------------------")
            
            # 1. Fetch exact context from vector DB using constraints
            db_context = fetch_from_vector_db(
                final_query=query_data.final_query, 
                doc_type=query_data.document_type, 
                clause_type=query_data.clause_type
            )
            print(f"-> Retrieved Context: \"{db_context}\"")
            
            # 2. Generator step: passing context to LLM to get answer
            qa_system_message = SystemMessage(content="You are a legal assistant. Answer the user's question strictly using the provided context. If the context does not contain the answer, say 'Insufficient context'. Do not use outside knowledge.")
            qa_human_message = HumanMessage(content=f"Context: {db_context}\n\nQuestion: {query_data.final_query}")
            
            answer_response = self.qa_llm.invoke([qa_system_message, qa_human_message])
            final_answer = answer_response.content
            print(f"-> RAG Answer: {final_answer}")
            
            # Store in output struct
            final_responses.append({
                "query": query_data.final_query,
                "document_type": query_data.document_type,
                "clause_type": query_data.clause_type,
                "retrieved_context": db_context,
                "answer": final_answer
            })
            
        return {
            "is_multi_query": analysis_result.is_multi_query,
            "results": final_responses
        }

if __name__ == "__main__":
    os.environ["OPENAI_API_KEY"] = "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"
    
    pipeline = LegalRAGPipeline(model_name="openai/gpt-4o")
    test_q = "payment terms lease. termination thing service doc how works, also explain liability clause"
    
    pipeline.execute_pipeline(test_q)
