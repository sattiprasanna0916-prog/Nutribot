
import json
import faiss
import numpy as np

from utils.config import (
    DATABASE_DIR,
    FAISS_INDEX_PATH,
    EMBEDDING_MODEL
)

from rag.embeddings import embedding_service


class NutriBotVectorStore:

    def __init__(self):

        self.chunks_path = DATABASE_DIR / "chunks.json"

        self.index_path = FAISS_INDEX_PATH

        # Use JSON instead of pickle for safer metadata storage
        self.metadata_path = DATABASE_DIR / "nutribot_metadata.json"

    def build_database(self):

        print("\nStarting NutriBot Vector Database Creation")
        print("=" * 50)

        # Check chunks file
        if not self.chunks_path.exists():
            raise FileNotFoundError(
                f"Chunks file not found: {self.chunks_path}"
            )

        # Load chunks
        with open(
            self.chunks_path,
            "r",
            encoding="utf-8"
        ) as file:

            chunks = json.load(file)

        if not chunks:
            raise ValueError("No text chunks found!")

        print("Total chunks loaded:", len(chunks))

        # Prepare texts
        texts = [
            chunk["text"] for chunk in chunks
        ]

        # Generate embeddings
        print("\nGenerating Hugging Face embeddings...")

        embeddings = embedding_service.generate_embeddings(
            texts
        )

        embeddings = np.asarray(
            embeddings,
            dtype=np.float32
        )

        # Validate embeddings
        if embeddings.shape[0] != len(chunks):
            raise ValueError(
                "Embedding count does not match chunk count!"
            )

        if embeddings.shape[1] != 384:
            raise ValueError(
                "Unexpected embedding dimensions!"
            )

        print("\nEmbeddings generated successfully!")
        print("Embedding shape:", embeddings.shape)

        # Create FAISS index
        print("\nCreating FAISS index...")

        dimension = embeddings.shape[1]

        index = faiss.IndexFlatIP(dimension)

        # Embeddings are already normalized
        index.add(embeddings)

        # Save FAISS index
        DATABASE_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        faiss.write_index(
            index,
            str(self.index_path)
        )

        # Save metadata
        with open(
            self.metadata_path,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                chunks,
                file,
                ensure_ascii=False,
                indent=2
            )

        print("\n" + "=" * 50)
        print("NUTRIBOT DATABASE CREATED SUCCESSFULLY!")
        print("=" * 50)

        print("Total vectors:", index.ntotal)
        print("Dimensions:", index.d)
        print("Metadata records:", len(chunks))

        print("\nFAISS Index:", self.index_path)
        print("Metadata:", self.metadata_path)

        return index


if __name__ == "__main__":

    vector_store = NutriBotVectorStore()

    vector_store.build_database()
