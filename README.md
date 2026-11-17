# 🛡️ CognitiveVault

**CognitiveVault** is a local RAG (Retrieval-Augmented Generation) assistant with a 3-column NotebookLM-style interface, multi-format file parsing, and a custom FastAPI backend powered by LlamaIndex, ChromaDB, and OpenRouter (`gpt-4o-mini`).

---

## 🌟 Key Features

* **Universal document upload:** PDF, Word (`.docx`), Excel (`.xlsx`), CSV, and Markdown files, indexed automatically in the background.
* **NotebookLM-inspired 3-column UI:** a warm-toned design (`#f7f3ed`) with a **Collections Sidebar**, a **Main Conversational Feed**, and a **Source Inspector** for real-time citation tracking.
* **Local vector persistence:** **ChromaDB** stores your knowledge base on your own machine.
* **Custom typography:** Google Fonts (**Montserrat**, **Cairo Play**, **Montserrat Alternates**).

---

## 🛠️ Tech Stack

* **Backend:** Python 3.12, FastAPI, Uvicorn, Pydantic
* **RAG & orchestration:** LlamaIndex, HuggingFace Embeddings (`BAAI/bge-small-en-v1.5`), OpenRouter (`gpt-4o-mini`)
* **Vector database:** ChromaDB (local persistence)
* **Frontend:** Tailwind CSS, HTML5, vanilla JavaScript
* **Document parsers:** `pypdf`, `python-docx`, `openpyxl`, `pandas`, `llama-index-readers-file`

---

## 📂 Project Structure

```text
cognitive-vault/
├── main.py              # FastAPI server & LlamaIndex RAG pipeline
├── requirements.txt     # Python dependencies
├── .env                 # API key configuration (not committed)
├── data/                # Uploaded source documents
├── chroma_db/           # Local ChromaDB vector store
└── static/
    └── index.html       # 3-column Tailwind CSS frontend
```

---

## 🚀 Getting Started

### 1. Clone and enter the project

```bash
git clone https://github.com/MuhammadZaeem500/Cognitive-Vault
cd cognitive-vault
```

### 2. Create a virtual environment and install dependencies

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip3 install -r requirements.txt
```

### 3. Configure environment variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openrouter_api_key_here
OPENAI_MODEL=gpt-4o-mini
```

### 4. Run the application

```bash
python3 main.py
```

The first run downloads the embedding model (`BAAI/bge-small-en-v1.5`), so it may take a minute.

### 5. Open in your browser

Go to http://127.0.0.1:8000.
