"""
NutriBot answer generator using Groq API.

This module keeps the existing FAISS retriever and replaces only Gemini
answer generation with Groq chat completions.
"""

import os
import time
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

from rag.retriever import retriever


PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = "openai/gpt-oss-120b"
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 2
MAX_CONTEXT_CHARS = 7000


if not GROQ_API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY was not found. Add it to D:\\NutriBot\\.env "
        "and restart the terminal."
    )

print("\nInitializing NutriBot Groq API...")
client = Groq(api_key=GROQ_API_KEY)
print(f"Groq API initialized successfully! Model: {MODEL_NAME}")


def _get_value(item, possible_keys, default=None):
    """Read a field from a dict or object using several likely field names."""
    if isinstance(item, dict):
        for key in possible_keys:
            if key in item and item[key] is not None:
                return item[key]
    else:
        for key in possible_keys:
            value = getattr(item, key, None)
            if value is not None:
                return value
    return default


class NutriBotGenerator:
    def __init__(self):
        self.client = client
        self.model_name = MODEL_NAME

    def _retrieve_passages(self, question):
        """Call the project's existing retriever without rebuilding the index."""
        if hasattr(retriever, "retrieve"):
            return retriever.retrieve(question)
        if callable(retriever):
            return retriever(question)
        raise TypeError(
            "Could not find a retrieve() method on rag.retriever.retriever. "
            "Please share rag/retriever.py so the interface can be matched."
        )

    def _format_context(self, results):
        context_parts = []
        sources = []

        for index, item in enumerate(results or [], start=1):
            text = _get_value(
                item,
                ("text", "content", "chunk", "page_content"),
                "",
            )
            page = _get_value(
                item,
                ("page", "page_number", "page_num", "source_page"),
                "Unknown",
            )
            score = _get_value(
                item,
                ("similarity", "score", "similarity_score"),
                None,
            )

            # Support metadata nested inside a result.
            metadata = _get_value(item, ("metadata",), {})
            if page == "Unknown" and metadata:
                page = _get_value(
                    metadata,
                    ("page", "page_number", "page_num", "source_page"),
                    "Unknown",
                )

            if not text:
                text = str(item)

            context_parts.append(
                f"[Passage {index} | Textbook page {page}]\n{str(text).strip()}"
            )

            source = {"page": page}
            if score is not None:
                source["similarity"] = score
            sources.append(source)

        context = "\n\n".join(context_parts)
        return context[:MAX_CONTEXT_CHARS], sources

    def _generate_with_retry(self, messages):
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                print(f"[Groq] Request attempt {attempt}/{MAX_RETRIES}...")
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=0.2,
                    max_completion_tokens=1200,
                )
                answer = response.choices[0].message.content
                if not answer:
                    raise RuntimeError("Groq returned an empty response.")
                print("[Groq] Response received.")
                return answer.strip()

            except Exception as exc:
                error_text = str(exc)
                temporary = any(
                    marker in error_text.lower()
                    for marker in (
                        "429", "500", "502", "503", "504",
                        "rate_limit", "timeout", "temporarily",
                        "service unavailable", "connection",
                    )
                )
                if temporary and attempt < MAX_RETRIES:
                    delay = RETRY_DELAY_SECONDS * (2 ** (attempt - 1))
                    print(f"[Groq] Temporary API issue: {exc}")
                    print(f"[Groq] Retrying in {delay} seconds...")
                    time.sleep(delay)
                    continue
                raise

    def generate_answer(self, question):
        print("\n[1] Starting FAISS retrieval...")
        try:
            results = self._retrieve_passages(question)
            if isinstance(results, tuple):
                results = results[0]
            if not isinstance(results, (list, tuple)):
                results = [results] if results else []

            print(f"[2] Retrieved {len(results)} passages.")
            if not results:
                return {
                    "answer": (
                        "I couldn't find relevant information in the "
                        "Human Nutrition textbook for that question. "
                        "Please try rephrasing it."
                    ),
                    "sources": [],
                    "context": "",
                }

            print("[3] Preparing textbook context...")
            context, sources = self._format_context(results)

            system_prompt = (
                "You are NutriBot, an educational assistant that answers "
                "questions using the supplied Human Nutrition textbook excerpts. "
                "Base factual claims on the excerpts. Do not invent textbook "
                "facts or page numbers. If the excerpts do not contain enough "
                "information, clearly say so. Explain clearly in simple language. "
                "Use short headings or bullet points when helpful. This is "
                "educational information, not personal medical advice."
            )
            user_prompt = (
                f"QUESTION:\n{question}\n\n"
                f"RETRIEVED TEXTBOOK EXCERPTS:\n{context}\n\n"
                "Answer the question using the excerpts above. Include the "
                "relevant textbook page number(s) in a short Sources line."
            )

            print(f"[4] Sending prompt to {self.model_name}...")
            answer = self._generate_with_retry(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
            )
            print("[5] Answer generation completed.")

            return {
                "answer": answer,
                "sources": sources,
                "context": context,
            }

        except Exception as exc:
            print(f"NutriBot generation error: {type(exc).__name__}: {exc}")
            return {
                "answer": (
                    "Sorry, NutriBot couldn't generate an answer because the "
                    f"Groq API request failed ({type(exc).__name__}). "
                    "Please check your API key, model availability, and account "
                    "rate limits, then try again."
                ),
                "sources": locals().get("sources", []),
                "context": locals().get("context", ""),
            }


generator = NutriBotGenerator()


if __name__ == "__main__":
    test_question = "What is the role of protein in the human body?"
    print("\nQUESTION:")
    print(test_question)

    result = generator.generate_answer(test_question)

    print("\n========== NUTRIBOT ANSWER ==========")
    print(result["answer"])

    print("\n========== SOURCES ==========")
    for source in result["sources"]:
        print(source)
