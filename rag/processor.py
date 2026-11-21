
import json
import pymupdf

from langchain_text_splitters import RecursiveCharacterTextSplitter

from utils.config import PDF_PATH, DATABASE_DIR


class NutritionPDFProcessor:

    def __init__(self):

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def process_pdf(self):

        if not PDF_PATH.exists():
            raise FileNotFoundError(
                f"Nutrition PDF not found: {PDF_PATH}"
            )

        print("Opening Human Nutrition textbook...")

        doc = pymupdf.open(str(PDF_PATH))

        chunks = []

        for page_number, page in enumerate(doc, start=1):

            text = page.get_text("text").strip()

            if not text:
                continue

            page_chunks = self.splitter.split_text(text)

            for chunk in page_chunks:

                if len(chunk.strip()) > 50:

                    chunks.append({
                        "text": chunk,
                        "page": page_number
                    })

        doc.close()

        # Save extracted chunks
        DATABASE_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        output_path = DATABASE_DIR / "chunks.json"

        with open(output_path, "w", encoding="utf-8") as file:

            json.dump(
                chunks,
                file,
                ensure_ascii=False,
                indent=2
            )

        print("\nPDF Processing Completed!")
        print("Total chunks:", len(chunks))
        print("Saved to:", output_path)

        return chunks


if __name__ == "__main__":

    processor = NutritionPDFProcessor()

    chunks = processor.process_pdf()
