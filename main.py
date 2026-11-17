import os
import shutil
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, List

import chromadb
import pandas as pd
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastembed import TextEmbedding
from llama_index.core import Document, Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.llms.openai_like import OpenAILike
from llama_index.vector_stores.chroma import ChromaVectorStore
from pydantic import BaseModel, PrivateAttr

load_dotenv()


# ---------- Embeddings (fastembed / ONNX, no PyTorch) ----------
class FastEmbedEmbedding(BaseEmbedding):
    """Minimal LlamaIndex wrapper around fastembed to keep memory use low."""

    _model: Any = PrivateAttr()

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", **kwargs: Any):
        super().__init__(model_name=model_name, **kwargs)
        self._model = TextEmbedding(model_name=model_name)

    @classmethod
    def class_name(cls) -> str:
        return "FastEmbedEmbedding"

    def _embed(self, texts: List[str]) -> List[List[float]]:
        return [vec.tolist() for vec in self._model.embed(texts)]

    def _get_query_embedding(self, query: str) -> List[float]:
        return self._embed([query])[0]

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)

    def _get_text_embedding(self, text: str) -> List[float]:
        return self._embed([text])[0]

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return self._embed(texts)


# ---------- Paths & config ----------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = BASE_DIR / "chroma_db"
STATIC_DIR = BASE_DIR / "static"
DATA_DIR.mkdir(exist_ok=True)

ALLOWED = {".pdf", ".docx", ".xlsx", ".csv", ".md"}
MAX_FILE_MB = 25

API_KEY = os.getenv("OPENAI_API_KEY", "")
API_BASE = os.getenv("OPENAI_API_BASE", "https://openrouter.ai/api/v1")
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
if "/" not in MODEL:                      # OpenRouter model ids look like "openai/gpt-4o-mini"
    MODEL = f"openai/{MODEL}"

# Chunks scoring below this are ignored. Tune it against your own questions.
SIMILARITY_CUTOFF = float(os.getenv("SIMILARITY_CUTOFF", "0.5"))
TOP_K = int(os.getenv("TOP_K", "5"))

if not API_KEY:
    print("WARNING: OPENAI_API_KEY is not set. Questions will fail until you add it to .env")

# ---------- LlamaIndex + Chroma ----------
Settings.embed_model = FastEmbedEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.llm = OpenAILike(
    model=MODEL,
    api_base=API_BASE,
    api_key=API_KEY or "missing-key",
    is_chat_model=True,
    context_window=128000,
    temperature=0.1,
    system_prompt=(
        "You answer questions using only the provided document excerpts. "
        "If the excerpts do not contain the answer, say you could not find it in the documents."
    ),
)
Settings.chunk_size = 512
Settings.chunk_overlap = 64

chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
collection = chroma_client.get_or_create_collection(
    "cognitive_vault", metadata={"hnsw:space": "cosine"}
)
vector_store = ChromaVectorStore(chroma_collection=collection)
index = VectorStoreIndex.from_vector_store(vector_store)
query_engine = index.as_query_engine(
    similarity_top_k=TOP_K,
    node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=SIMILARITY_CUTOFF)],
)

index_lock = threading.Lock()


# ---------- Helpers ----------
def load_documents(path: Path) -> list[Document]:
    """Read a file into LlamaIndex Documents tagged with its file name."""
    if path.suffix.lower() == ".xlsx":
        sheets = pd.read_excel(path, sheet_name=None)
        docs = [
            Document(text=f"Sheet: {name}\n{df.to_csv(index=False)}")
            for name, df in sheets.items()
            if not df.empty
        ]
    else:
        docs = SimpleDirectoryReader(input_files=[str(path)]).load_data()

    docs = [d for d in docs if d.text and d.text.strip()]
    for d in docs:
        d.metadata = {"file_name": path.name}
    return docs


def remove_from_index(file_name: str) -> None:
    collection.delete(where={"file_name": file_name})


def index_file(path: Path) -> None:
    docs = load_documents(path)
    if not docs:
        raise ValueError("no readable text found (scanned PDF or empty file?)")
    with index_lock:
        remove_from_index(path.name)      # re-uploading a file replaces its old vectors
        for d in docs:
            index.insert(d)


def indexed_names() -> set[str]:
    data = collection.get(include=["metadatas"])
    return {m["file_name"] for m in (data["metadatas"] or []) if m and m.get("file_name")}


def disk_names() -> set[str]:
    return {p.name for p in DATA_DIR.iterdir() if p.is_file() and p.suffix.lower() in ALLOWED}


def sync_index() -> None:
    """Make the vector store match the data/ folder."""
    on_disk, indexed = disk_names(), indexed_names()
    for name in indexed - on_disk:        # file was deleted -> drop its vectors
        remove_from_index(name)
        print(f"Removed stale vectors for: {name}")
    for name in on_disk - indexed:        # file exists but was never indexed
        try:
            index_file(DATA_DIR / name)
            print(f"Indexed: {name}")
        except Exception as e:
            print(f"Could not index {name}: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    sync_index()
    yield


# ---------- App ----------
app = FastAPI(title="CognitiveVault", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class QueryRequest(BaseModel):
    prompt: str


@app.get("/")
def home():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/files")
def list_files():
    return {"files": sorted(disk_names())}


@app.post("/api/upload")
def upload(files: list[UploadFile] = File(...)):
    errors, saved = [], 0
    for f in files:
        name = Path(f.filename or "").name
        if Path(name).suffix.lower() not in ALLOWED:
            errors.append(f"{name or 'file'}: unsupported file type")
            continue

        dest = DATA_DIR / name
        with dest.open("wb") as out:
            shutil.copyfileobj(f.file, out)

        if dest.stat().st_size > MAX_FILE_MB * 1024 * 1024:
            dest.unlink(missing_ok=True)
            errors.append(f"{name}: larger than {MAX_FILE_MB} MB")
            continue

        try:
            index_file(dest)
            saved += 1
        except Exception as e:
            dest.unlink(missing_ok=True)
            remove_from_index(name)
            errors.append(f"{name}: {e}")

    if errors and saved == 0:
        raise HTTPException(status_code=400, detail="; ".join(errors))
    return {"ok": True, "indexed": saved, "errors": errors}


@app.delete("/api/files/{file_name}")
def delete_file(file_name: str):
    name = Path(file_name).name           # blocks ../ path tricks
    path = DATA_DIR / name
    if not path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    path.unlink()
    with index_lock:
        remove_from_index(name)
    return {"ok": True}


@app.post("/api/query")
def query(req: QueryRequest):
    prompt = req.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Empty question")

    if collection.count() == 0:
        return {
            "answer": "Your vault is empty. Upload a document first, then ask me about it.",
            "sources": [],
        }

    try:
        response = query_engine.query(prompt)
    except Exception as e:
        print(f"Query failed: {e}")
        raise HTTPException(
            status_code=502,
            detail="The language model request failed. Check your API key and model name.",
        )

    nodes = response.source_nodes
    if not nodes:
        return {
            "answer": "I couldn't find anything relevant to that in your documents. "
                      "Try rephrasing, or upload a file that covers it.",
            "sources": [],
        }

    sources = [
        {
            "file_name": n.node.metadata.get("file_name", "unknown"),
            "score": float(n.score or 0.0),
            "text": n.node.get_content()[:1200],
        }
        for n in nodes
    ]
    return {"answer": str(response).strip(), "sources": sources}


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    host = "0.0.0.0" if os.getenv("PORT") else "127.0.0.1"
    uvicorn.run(app, host=host, port=port)