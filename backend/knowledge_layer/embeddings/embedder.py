from sentence_transformers import SentenceTransformer
import os
import logging

# Configure local logger for this submodule
logger = logging.getLogger("knowledge_layer.embeddings")

class BGEEmbedder:
    def __init__(self, model_name="BAAI/bge-small-en-v1.5"):
        self.model_name = model_name
        self._model = None
        logger.info(f"Embedder initialized (model will be lazy-loaded on first request)")

    @property
    def model(self):
        """Lazy-loads the model only when needed."""
        if self._model is None:
            logger.info(f"Loading embedding model: {self.model_name}...")
            try:
                self._model = SentenceTransformer(self.model_name)
                logger.info("Embedding model loaded successfully.")
            except Exception as e:
                logger.error(f"Failed to load embedding model: {e}")
                raise
        return self._model

    def embed_text(self, text):
        """Generates a single embedding for the provided text."""
        if not text or not text.strip():
            return None
        # Accessing self.model triggers the lazy load
        return self.model.encode(text, normalize_embeddings=True).tolist()

    def embed_batch(self, texts):
        """Generates embeddings for a batch of texts."""
        logger.info(f"Generating embeddings for batch of {len(texts)} items...")
        return self.model.encode(texts, normalize_embeddings=True).tolist()

# Singleton instance
embedder = BGEEmbedder()
