import os
import re
import shutil
import threading
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, List

import chromadb
import jwt
import pandas as pd
import uvicorn
from database import (
    SessionLocal,
    UserDB,
    get_password_hash,
    init_db,
    verify_password,
)
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastembed import TextEmbedding
from llama_index.core import (
    Document,
    PromptTemplate,
    Settings,
    SimpleDirectoryReader,
    VectorStoreIndex,
)
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters
from llama_index.llms.openai_like import OpenAILike
from llama_index.vector_stores.chroma import ChromaVectorStore
from pydantic import BaseModel, PrivateAttr
from sqlalchemy.orm import Session

load_dotenv()
init_db()

SECRET_KEY = os.getenv(
    "JWT_SECRET_KEY", "super-secret-key-change-in-production"
)
ALGORITHM = "HS256"

# Common disposable/fake email domains to block instantly
DISPOSABLE_DOMAINS = {
    "mailinator.com",
    "10minutemail.com",
    "tempmail.com",
    "guerrillamail.com",
    "sharklasers.com",
    "trashmail.com",
    "yopmail.com",
    "getnada.com",
    "dispostable.com",
    "fakeinbox.com",
}


def is_valid_email(email: str) -> bool:
  pattern = r"^[\w\.-]+@([\w-]+\.)+[\w-]{2,}$"
  if not re.match(pattern, email):
    return False
  domain = email.split("@")[-1].lower()
  if domain in DISPOSABLE_DOMAINS:
    return False
  return True


def get_db():
  db = SessionLocal()
  try:
    yield db
  finally:
    db.close()


def create_access_token(data: dict):
  to_encode = data.copy()
  expire = datetime.utcnow() + timedelta(days=7)
  to_encode.update({"exp": expire})
  return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def get_current_user(
    authorization: str = Header(None), db: Session = Depends(get_db)
):
  if not authorization or not authorization.startswith("Bearer "):
    raise HTTPException(status_code=401, detail="Not authenticated")
  token = authorization.split(" ")[1]
  try:
    payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    email: str = payload.get("sub")
    if email is None:
      raise HTTPException(status_code=401, detail="Invalid token")
  except jwt.PyJWTError:
    raise HTTPException(status_code=401, detail="Invalid token")

  user = db.query(UserDB).filter(UserDB.email == email).first()
  if not user:
    raise HTTPException(status_code=401, detail="User not found")
  return user


# ---------- Embeddings ----------
class FastEmbedEmbedding(BaseEmbedding):
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
if "/" not in MODEL:
  MODEL = f"openai/{MODEL}"

SIMILARITY_CUTOFF = float(os.getenv("SIMILARITY_CUTOFF", "0.3"))
TOP_K = int(os.getenv("TOP_K", "5"))

NOT_FOUND_MESSAGE = (
    "I couldn't find anything about that in your documents. Try asking about"
    " something specific from them."
)

QA_TEMPLATE = PromptTemplate(
    "You are a helpful assistant answering questions about the user's uploaded"
    " documents.\nExcerpts from the documents are"
    " below.\n---------------------\n{context_str}\n---------------------\nUse"
    " the excerpts to answer the question. If the user asks what a document is"
    " about or what it contains, summarize the excerpts. Do not use outside"
    " knowledge. If the excerpts clearly do not help, reply with exactly:"
    " NOT_FOUND\nQuestion: {query_str}\nAnswer: "
)

# ---------- LlamaIndex + Chroma ----------
Settings.embed_model = FastEmbedEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.llm = OpenAILike(
    model=MODEL,
    api_base=API_BASE,
    api_key=API_KEY or "missing-key",
    is_chat_model=True,
    context_window=128000,
    temperature=0.1,
)
Settings.chunk_size = 512
Settings.chunk_overlap = 64

chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
collection = chroma_client.get_or_create_collection(
    "cognitive_vault", metadata={"hnsw:space": "cosine"}
)
vector_store = ChromaVectorStore(chroma_collection=collection)
index = VectorStoreIndex.from_vector_store(vector_store)

index_lock = threading.Lock()


# ---------- Schemas & Auth Routes ----------
class UserSignupSchema(BaseModel):
  name: str
  email: str
  password: str


class UserLoginSchema(BaseModel):
  email: str
  password: str


class QueryRequest(BaseModel):
  prompt: str


app = FastAPI(title="CognitiveVault")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.post("/api/auth/signup")
def signup(payload: UserSignupSchema, db: Session = Depends(get_db)):
  email = payload.email.strip().lower()
  if not is_valid_email(email):
    raise HTTPException(
        status_code=400,
        detail="Invalid or temporary/fake email address not allowed.",
    )

  if not payload.name or len(payload.name.strip()) < 2:
    raise HTTPException(
        status_code=400, detail="Please provide a valid name."
    )

  existing = db.query(UserDB).filter(UserDB.email == email).first()
  if existing:
    raise HTTPException(status_code=400, detail="Email already registered")

  user_id = str(uuid.uuid4())
  hashed_pw = get_password_hash(payload.password)
  new_user = UserDB(
      id=user_id, name=payload.name.strip(), email=email, hashed_password=hashed_pw
  )
  db.add(new_user)
  db.commit()

  # Create user specific directory
  (DATA_DIR / user_id).mkdir(parents=True, exist_ok=True)
  return {"ok": True, "message": "Account created successfully"}


@app.post("/api/auth/login")
def login(payload: UserLoginSchema, db: Session = Depends(get_db)):
  user = db.query(UserDB).filter(UserDB.email == payload.email.strip().lower()).first()
  if not user or not verify_password(payload.password, user.hashed_password):
    raise HTTPException(status_code=400, detail="Incorrect email or password")

  access_token = create_access_token(data={"sub": user.email})
  return {"access_token": access_token, "token_type": "bearer"}


@app.get("/api/auth/me")
def get_me(current_user: UserDB = Depends(get_current_user)):
  return {"name": current_user.name, "email": current_user.email}


# ---------- Helpers & Endpoints ----------
def get_user_data_dir(user_id: str) -> Path:
  d = DATA_DIR / user_id
  d.mkdir(parents=True, exist_ok=True)
  return d


def load_documents(path: Path, user_id: str) -> list[Document]:
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
    d.metadata = {"file_name": path.name, "user_id": user_id}
  return docs


def remove_from_index(file_name: str, user_id: str) -> None:
  collection.delete(
      where={"$and": [{"file_name": file_name}, {"user_id": user_id}]}
  )


def index_file(path: Path, user_id: str) -> None:
  docs = load_documents(path, user_id)
  if not docs:
    raise ValueError("no readable text found")
  with index_lock:
    remove_from_index(path.name, user_id)
    for d in docs:
      index.insert(d)


def indexed_names(user_id: str) -> set[str]:
  data = collection.get(where={"user_id": user_id}, include=["metadatas"])
  return {
      m["file_name"] for m in (data["metadatas"] or []) if m and m.get("file_name")
  }


def disk_names(user_id: str) -> set[str]:
  user_dir = get_user_data_dir(user_id)
  return {
      p.name
      for p in user_dir.iterdir()
      if p.is_file() and p.suffix.lower() in ALLOWED
  }


@app.get("/")
def home():
  return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/files")
def list_files(current_user: UserDB = Depends(get_current_user)):
  return {"files": sorted(disk_names(current_user.id))}


@app.post("/api/upload")
def upload(
    files: list[UploadFile] = File(...),
    current_user: UserDB = Depends(get_current_user),
):
  user_dir = get_user_data_dir(current_user.id)
  errors, saved = [], 0
  for f in files:
    name = Path(f.filename or "").name
    if Path(name).suffix.lower() not in ALLOWED:
      errors.append(f"{name or 'file'}: unsupported file type")
      continue

    dest = user_dir / name
    with dest.open("wb") as out:
      shutil.copyfileobj(f.file, out)

    if dest.stat().st_size > MAX_FILE_MB * 1024 * 1024:
      dest.unlink(missing_ok=True)
      errors.append(f"{name}: larger than {MAX_FILE_MB} MB")
      continue

    try:
      index_file(dest, current_user.id)
      saved += 1
    except Exception as e:
      dest.unlink(missing_ok=True)
      remove_from_index(name, current_user.id)
      errors.append(f"{name}: {e}")

  if errors and saved == 0:
    raise HTTPException(status_code=400, detail="; ".join(errors))
  return {"ok": True, "indexed": saved, "errors": errors}


@app.delete("/api/files/{file_name}")
def delete_file(
    file_name: str, current_user: UserDB = Depends(get_current_user)
):
  name = Path(file_name).name
  path = get_user_data_dir(current_user.id) / name
  if not path.exists():
    raise HTTPException(status_code=404, detail="File not found")
  path.unlink()
  with index_lock:
    remove_from_index(name, current_user.id)
  return {"ok": True}


@app.post("/api/query")
def query(
    req: QueryRequest, current_user: UserDB = Depends(get_current_user)
):
  prompt = req.prompt.strip()
  if not prompt:
    raise HTTPException(status_code=400, detail="Empty question")

  user_filters = MetadataFilters(
      filters=[MetadataFilter(key="user_id", value=current_user.id)]
  )

  user_query_engine = index.as_query_engine(
      similarity_top_k=TOP_K,
      text_qa_template=QA_TEMPLATE,
      node_postprocessors=[
          SimilarityPostprocessor(similarity_cutoff=SIMILARITY_CUTOFF)
      ],
      filters=user_filters,
  )

  try:
    response = user_query_engine.query(prompt)
  except Exception as e:
    raise HTTPException(status_code=502, detail="The language model request failed.")

  nodes = [
      n
      for n in response.source_nodes
      if n.node.metadata.get("user_id") == current_user.id
  ]
  answer = str(response).strip()

  if not nodes or not answer or "NOT_FOUND" in answer.upper():
    return {"answer": NOT_FOUND_MESSAGE, "sources": []}

  sources = [
      {
          "file_name": n.node.metadata.get("file_name", "unknown"),
          "score": float(n.score or 0.0),
          "text": n.node.get_content()[:1200],
      }
      for n in nodes
  ]
  return {"answer": answer, "sources": sources}


if __name__ == "__main__":
  port = int(os.getenv("PORT", "8000"))
  host = "0.0.0.0" if os.getenv("PORT") else "127.0.0.1"
  uvicorn.run(app, host=host, port=port)