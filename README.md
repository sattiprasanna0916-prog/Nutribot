# NutriBot – AI-Powered Human Nutrition Assistant

## Project Overview
NutriBot is an AI-powered nutrition assistant that uses Retrieval-Augmented Generation (RAG) to answer nutrition-related questions based on the Human Nutrition – 2020 Edition textbook.

## Technologies Used
- Python
- LangChain
- FAISS
- Hugging Face Sentence Transformers
- MiniLM (all-MiniLM-L6-v2)
- Groq API
- Gradio
- PyMuPDF

## System Architecture
1. Extract text from the nutrition textbook using PyMuPDF.
2. Split the extracted text into smaller chunks using LangChain.
3. Generate semantic embeddings using MiniLM.
4. Store embeddings in a FAISS vector database.
5. Retrieve relevant passages based on user questions.
6. Generate answers using the Groq API and retrieved context.
7. Display answers and source references through Gradio.

## Features
- AI-based nutrition question answering
- Semantic document search
- Textbook-based responses
- Source page references
- Interactive chatbot interface

## How to Run

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/NutriBot.git
cd NutriBot
```

### 2. Create a virtual environment
```bash
python -m venv .venv
```

### 3. Activate it (Windows)
```powershell
.venv\Scripts\activate
```

### 4. Install dependencies
```bash
pip install -r requirements.txt
```

### 5. Configure the API key
Create a `.env` file in the project root:
```env
GROQ_API_KEY=your_groq_api_key
```

### 6. Add the textbook
Place `Human-Nutrition-2020-Edition.pdf` inside the `data` folder.

### 7. Run the application
```bash
python app.py
```

Open http://127.0.0.1:7860 in your browser.

## Disclaimer
NutriBot is an educational information assistant and does not provide medical diagnosis or personalized medical advice.