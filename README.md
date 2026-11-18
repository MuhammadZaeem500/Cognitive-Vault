# 🛡️ CognitiveVault

**CognitiveVault** is a lightweight, local-first, universal RAG (Retrieval-Augmented Generation) assistant. It allows users to securely upload, store, index, and query across multiple document formats (`.pdf`, `.docx`, `.xlsx`, `.csv`, `.md`) with strict per-user data isolation, powered by FastAPI, LlamaIndex, ChromaDB, FastEmbed, and OpenRouter (`gpt-4o-mini`).

---

## 🚀 Key Features

- **Strict User Data Isolation**: Powered by SQLAlchemy-backed user management and LlamaIndex metadata filtering (`user_id`), ensuring users can only search and view their own uploaded files.
- **Local Embeddings**: Uses `fastembed` (`BAAI/bge-small-en-v1.5`) for running efficient, high-performance vector embeddings locally on your machine without external embedding API costs.
- **Robust Authentication**: Secure JWT authentication paired with PBKDF2-HMAC-SHA256 password hashing (optimized for Python 3.12 compatibility) and automatic disposable/fake email blocking.
- **Multi-Format Document Support**: Seamlessly parses and chunks PDFs, Word documents, Excel workbooks (sheet-by-sheet), CSVs, and Markdown files.
- **Modern Three-Column UI**: Styled with Tailwind CSS featuring a collections sidebar, a conversational chat feed with citation badges, and a real-time Source Inspector panel.

---

## 🛠️ Tech Stack

- **Backend**: Python, FastAPI, Uvicorn, SQLAlchemy, PyJWT
- **RAG & Search**: LlamaIndex, ChromaDB (Persistent local vector store), FastEmbed
- **AI Model**: OpenRouter API (`gpt-4o-mini`)
- **Frontend**: HTML5, Vanilla JavaScript, Tailwind CSS

---

## ⚙️ Installation & Setup

### 1. Clone or Navigate to the Project Directory
```bash
cd cognitive-vault
```

### 2. Create and Activate a Virtual Environment
```python
python3 -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install fastapi uvicorn sqlalchemy pyjwt pandas python-dotenv chromadb llama-index fastembed
```

### 4. Configure Environment Variables
Create a `.env` file in the root directory with your configuration:
```env
OPENAI_API_KEY=your_openrouter_api_key_here
OPENAI_MODEL=gpt-4o-mini
```

---

## ▶️ Running the Application

Start the FastAPI development server:
```bash
python3 main.py
```

Open your browser and navigate to:
👉 **`http://127.0.0.1:8000`**

---

## 📂 Project Structure

- `main.py` — FastAPI application server, routing, authentication middleware, and LlamaIndex RAG query pipeline.
- `database.py` — SQLAlchemy configuration, user schema models, and password hashing logic.
- `static/` — Frontend assets (`index.html` and custom UI stylesheets).
- `data/` — Local user document storage directory (segmented by `user_id`).
- `chroma_db/` — Persistent Chroma vector database storage.