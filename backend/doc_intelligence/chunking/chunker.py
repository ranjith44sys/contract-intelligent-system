import tiktoken
from ..config import logger

class Chunker:
    def __init__(self, chunk_size=700, overlap=100, model="gpt-4o"):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.encoding = tiktoken.encoding_for_model(model)

    def create_chunks(self, text, metadata=None):
        """Splits text into token-based chunks with overlap."""
        tokens = self.encoding.encode(text)
        total_tokens = len(tokens)
        chunks = []
        
        start = 0
        while start < total_tokens:
            end = min(start + self.chunk_size, total_tokens)
            chunk_tokens = tokens[start:end]
            chunk_text = self.encoding.decode(chunk_tokens)
            
            chunk_id = f"CHUNK_{len(chunks) + 1:04d}"
            if metadata and "clause_id" in metadata:
                chunk_id = f"{metadata['clause_id']}_{chunk_id}"

            chunks.append({
                "chunk_id": chunk_id,
                "text": chunk_text,
                "token_count": len(chunk_tokens),
                "metadata": metadata or {}
            })
            
            if end == total_tokens:
                break
            
            start = end - self.overlap
            
        return chunks

    def chunk_clauses(self, clauses_graph):
        """Processes a full clause graph and adds chunks to each clause."""
        logger.info(f"Chunking {len(clauses_graph)} clauses...")
        for clause in clauses_graph:
            meta = {
                "clause_id": clause["clause_id"],
                "contract_type": clause["contract_type"],
                "page_number": clause["page_number"]
            }
            clause["chunks"] = self.create_chunks(clause["text"], meta)
            
        return clauses_graph
