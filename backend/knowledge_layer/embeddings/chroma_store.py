import chromadb
from chromadb.config import Settings
import os
import logging

logger = logging.getLogger("knowledge_layer.storage")

class ChromaStore:
    def __init__(self, persist_directory):
        self.persist_directory = persist_directory
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection_name = "legal_documents"
        
        # Initialize or get collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"} # BGE models work well with cosine similarity
        )
        logger.info(f"Initialized ChromaDB collection: {self.collection_name} at {persist_directory}")

    def add_documents(self, ids, embeddings, documents, metadatas):
        """Adds vectorized documents to the ChromaDB collection."""
        logger.info(f"Adding {len(ids)} items to ChromaDB...")
        try:
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )
            logger.info("Successfully added documents to ChromaDB.")
        except Exception as e:
            logger.error(f"Failed to add documents to ChromaDB: {e}")
            raise

    def reset_collection(self):
        """Clears all data from the collection while preserving schema."""
        logger.warning(f"Resetting ChromaDB collection: {self.collection_name}")
        try:
            # Delete all documents (using a catch-all filter if needed, 
            # or just delete the collection and recreate)
            # collection.delete() with no arguments deletes all
            self.collection.delete()
            logger.info("ChromaDB collection cleared successfully.")
        except Exception as e:
            logger.error(f"Failed to reset ChromaDB: {e}")

    def query(self, query_embeddings, n_results=5, where=None):
        """Performs a vector search query."""
        return self.collection.query(
            query_embeddings=query_embeddings,
            n_results=n_results,
            where=where
        )

# Configuration for default instance
DEFAULT_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output", "chroma_db")
chroma_store = ChromaStore(DEFAULT_DB_PATH)
