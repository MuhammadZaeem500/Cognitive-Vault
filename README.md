# 🛡️ CognitiveVault

**CognitiveVault** is a local RAG (Retrieval-Augmented Generation) assistant web application featuring a 3-column NotebookLM-style interface, universal multi-format file attachment parsing, and a custom FastAPI backend powered by LlamaIndex, ChromaDB, and OpenRouter (`gpt-4o-mini`).

---

## 🌟 Key Features

* **Universal Document Attachments:** Upload PDFs, Word (`.docx`), Excel (`.xlsx`), CSVs, and Markdown files with automated background indexing.
* **NotebookLM-Inspired 3-Column UI:** A clean, warm-toned aesthetic featuring a **Collections Sidebar**, **Main Conversational Feed**, and a **Source Inspector** for real-time citation tracking.
* **Custom Typography:** Styled with Google Fonts (**Montserrat**, **Cairo Play**, and **Montserrat Alternates**).
* **Local Vector Persistence:** Powered by **ChromaDB** to securely persist and query your local knowledge base.

---

## 🛠️ Tech Stack

* **Backend:** Python 3.12, FastAPI, Uvicorn, Gunicorn, Pydantic
* **RAG & Orchestration:** LlamaIndex, HuggingFace Embeddings (`BAAI/bge-small-en-v1.5`), OpenRouter (`gpt-4o-mini`)
* **Vector Database:** ChromaDB (local persistence)
* **Frontend:** Tailwind CSS, HTML5, vanilla JavaScript
* **Document Parsers:** `pypdf`, `python-docx`, `openpyxl`, `pandas`, `llama-index-readers-file`

---

## 📂 Project Directory Structure

```text
cognitive-vault/
├── main.py              # FastAPI server & LlamaIndex RAG pipeline
├── requirements.txt     # Python dependencies & production workers
├── .env                 # API keys configuration (not committed)
├── data/                # Uploaded multi-format source documents
├── chroma_db/           # Local ChromaDB vector store directory
└── static/
    └── index.html       # 3-column Tailwind CSS frontend interface
```

---

## 🚀 Local Development Setup

### 1. Clone & navigate to the repository

```bash
git clone https://github.com/MuhammadZaeem500/Cognitive-Vault.git
cd Cognitive-Vault
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

Create a `.env` file in the root directory and add your OpenRouter credentials:

```env
OPENAI_API_KEY=your_openrouter_api_key_here
OPENAI_MODEL=gpt-4o-mini
```

### 4. Run the application

```bash
python3 main.py
```

### 5. Open in browser

Navigate to `http://127.0.0.1:8000` to interact with your CognitiveVault assistant.

---

## ☁️ Deployment on Render

This project is configured for production deployment on [Render](https://render.com/):

1. Create a new **Web Service** on Render connected to your `Cognitive-Vault` GitHub repository.
2. Use the following configuration:
   * **Runtime:** `Python 3`
   * **Build Command:** `pip3 install -r requirements.txt`
   * **Start Command:**
     ```bash
     gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:$PORT
     ```
3. Add your environment variables (`OPENAI_API_KEY` and `OPENAI_MODEL`) under the Render **Environment** tab and deploy.