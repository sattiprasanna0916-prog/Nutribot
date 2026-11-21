
from pathlib import Path

# Project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Main directories
DATA_DIR = BASE_DIR / "data"
DATABASE_DIR = BASE_DIR / "database"
RAG_DIR = BASE_DIR / "rag"
UI_DIR = BASE_DIR / "ui"

# Textbook
PDF_PATH = DATA_DIR / "Human-Nutrition-2020-Edition.pdf"

# FAISS database
FAISS_INDEX_PATH = DATABASE_DIR / "nutribot_faiss.index"
METADATA_PATH = DATABASE_DIR / "nutribot_metadata.json"

# Hugging Face models
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GENERATION_MODEL = "google/flan-t5-small"

# RAG settings
TOP_K = 4
SIMILARITY_THRESHOLD = 0.30
MAX_CONTEXT_CHARS = 1600

# Application settings
APP_NAME = "NutriBot"
APP_DESCRIPTION = "AI-Powered Human Nutrition Assistant"

# Device configuration
DEVICE = "cpu"

print("NutriBot configuration loaded!")
