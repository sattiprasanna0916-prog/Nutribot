
from sentence_transformers import SentenceTransformer

from utils.config import EMBEDDING_MODEL, DEVICE


class EmbeddingService:

    def __init__(self):
        """Load the Hugging Face embedding model."""

        print("Loading NutriBot embedding model...")

        self.model = SentenceTransformer(
            EMBEDDING_MODEL,
            device=DEVICE
        )

        self.dimension = self.model.get_sentence_embedding_dimension()

        print("Embedding model loaded successfully!")
        print("Model:", EMBEDDING_MODEL)
        print("Embedding dimensions:", self.dimension)

    def generate_embedding(self, text):
        """Generate embedding for a single text."""

        embedding = self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        return embedding

    def generate_embeddings(self, texts):
        """Generate embeddings for multiple texts."""

        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        return embeddings


# Create reusable embedding service
embedding_service = EmbeddingService()
