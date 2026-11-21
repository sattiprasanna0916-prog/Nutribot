
import json
import faiss
import numpy as np

from utils.config import (
    FAISS_INDEX_PATH,
    METADATA_PATH,
    TOP_K,
    SIMILARITY_THRESHOLD
)

from rag.embeddings import embedding_service


class NutriBotRetriever:

    def __init__(self):

        print("Loading NutriBot knowledge base...")

        # Verify database files
        if not FAISS_INDEX_PATH.exists():
            raise FileNotFoundError(
                f"FAISS index not found: {FAISS_INDEX_PATH}"
            )

        if not METADATA_PATH.exists():
            raise FileNotFoundError(
                f"Metadata not found: {METADATA_PATH}"
            )

        # Load FAISS index
        self.index = faiss.read_index(
            str(FAISS_INDEX_PATH)
        )

        # Load JSON metadata
        with open(
            METADATA_PATH,
            "r",
            encoding="utf-8"
        ) as file:

            self.metadata = json.load(file)

        # Validate database
        if self.index.ntotal != len(self.metadata):
            raise ValueError(
                "FAISS and metadata counts do not match!"
            )

        if self.index.d != embedding_service.dimension:
            raise ValueError(
                "Embedding dimension mismatch!"
            )

        self.threshold = SIMILARITY_THRESHOLD

        print("NutriBot knowledge base loaded!")
        print("Total vectors:", self.index.ntotal)
        print("Metadata records:", len(self.metadata))
        print("Dimensions:", self.index.d)

    def retrieve(self, query, top_k=None):

        if not query or not query.strip():
            return []

        top_k = top_k or TOP_K

        top_k = min(
            max(1, top_k),
            self.index.ntotal
        )

        # Generate question embedding
        query_vector = embedding_service.generate_embedding(
            query.strip()
        )

        query_vector = np.asarray(
            [query_vector],
            dtype=np.float32
        )

        # FAISS similarity search
        scores, indices = self.index.search(
            query_vector,
            top_k
        )

        results = []

        for score, idx in zip(scores[0], indices[0]):

            if idx == -1:
                continue

            score = float(score)

            metadata = self.metadata[idx]

            results.append({
                "text": metadata["text"],
                "page": metadata["page"],
                "similarity": round(score, 4)
            })

        # Preserve unfiltered results for diagnostics.
        filtered_results = [
            result for result in results
            if result["similarity"] >= self.threshold
        ]

        return filtered_results

    def diagnose(self, query, top_k=5):
        """Display retrieval results for debugging."""

        results = self.retrieve(query, top_k)

        print("\nNUTRIBOT RETRIEVAL DIAGNOSTICS")
        print("=" * 55)

        print("Question:", query)
        print("Retrieved passages:", len(results))

        for i, result in enumerate(results, start=1):

            print(f"\nResult {i}")
            print("-" * 40)

            print("PDF Page:", result["page"])
            print("Similarity:", result["similarity"])
            print("Text:", result["text"][:300])


# Create reusable retriever
retriever = NutriBotRetriever()


if __name__ == "__main__":

    retriever.diagnose(
        "What is the role of protein in the human body?"
    )
