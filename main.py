import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from llama_index.core import SimpleDirectoryReader, Settings, VectorStoreIndex, StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

load_dotenv()

app = FastAPI(title="CognitiveVault API")

os.makedirs("./data", exist_ok=True)
os.makedirs("static", exist_ok=True)

# Mount static folder for HTML/CSS frontend
app.mount("/static", StaticFiles(directory="static"), name="static")

def get_query_engine():
    openrouter_api_key = os.getenv("OPENAI_API_KEY")
    if not openrouter_api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not configured in environment.")

    Settings.llm = OpenAI(
        model="gpt-4o-mini",
        temperature=0.1,
        api_key=openrouter_api_key,
        api_base="https://openrouter.ai/api/v1"
    )
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

    db = chromadb.PersistentClient(path="./chroma_db")
    chroma_collection = db.get_or_create_collection("cognitive_vault_kb")

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
    return index.as_query_engine(similarity_top_k=3)

class QueryRequest(BaseModel):
    prompt: str

@app.get("/")
def serve_frontend():
    return FileResponse("static/index.html")

@app.get("/api/files")
def list_files():
    files = [f for f in os.listdir("./data") if not f.startswith('.')]
    return {"files": files}

@app.post("/api/upload")
async def upload_files(files: list[UploadFile] = File(...)):
    try:
        saved_files = []
        for file in files:
            # Only save supported document extensions
            ext = os.path.splitext(file.filename)[1].lower()
            if ext in [".pdf", ".docx", ".xlsx", ".csv", ".txt", ".md"]:
                file_path = os.path.join("./data", file.filename)
                with open(file_path, "wb") as buffer:
                    buffer.write(await file.read())
                saved_files.append(file.filename)
        
        if not saved_files:
            return {"status": "ignored", "message": "No supported document formats found."}
        
        # Trigger background auto-indexing safely
        db = chromadb.PersistentClient(path="./chroma_db")
        chroma_collection = db.get_or_create_collection("cognitive_vault_kb")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        documents = SimpleDirectoryReader(
            "./data", 
            recursive=True,
            required_exts=[".pdf", ".docx", ".xlsx", ".csv", ".txt", ".md"]
        ).load_data()
        
        VectorStoreIndex.from_documents(documents, storage_context=storage_context)
        
        return {"status": "success", "message": "Files uploaded and indexed successfully."}
    except Exception as e:
        print(f"Upload/Indexing Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    try:
        for file in files:
            file_path = os.path.join("./data", file.filename)
            with open(file_path, "wb") as buffer:
                buffer.write(await file.read())
        
        # Trigger background auto-indexing safely
        db = chromadb.PersistentClient(path="./chroma_db")
        chroma_collection = db.get_or_create_collection("cognitive_vault_kb")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        # Use SimpleDirectoryReader with safe file filtering or standard loader
        documents = SimpleDirectoryReader(
            "./data", 
            recursive=True,
            required_exts=[".pdf", ".docx", ".xlsx", ".csv", ".txt", ".md"]
        ).load_data()
        
        VectorStoreIndex.from_documents(documents, storage_context=storage_context)
        
        return {"status": "success", "message": "Files uploaded and indexed successfully."}
    except Exception as e:
        print(f"Upload/Indexing Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    try:
        for file in files:
            file_path = os.path.join("./data", file.filename)
            with open(file_path, "wb") as buffer:
                buffer.write(await file.read())
        
        # Trigger background auto-indexing
        db = chromadb.PersistentClient(path="./chroma_db")
        chroma_collection = db.get_or_create_collection("cognitive_vault_kb")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        documents = SimpleDirectoryReader("./data", recursive=True).load_data()
        VectorStoreIndex.from_documents(documents, storage_context=storage_context)
        
        return {"status": "success", "message": "Files uploaded and indexed successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/query")
def query_vault(req: QueryRequest):
    try:
        engine = get_query_engine()
        response = engine.query(req.prompt)
        
        sources = []
        if response.source_nodes:
            for node in response.source_nodes:
                sources.append({
                    "file_name": node.metadata.get("file_name", "Unknown"),
                    "score": getattr(node, "score", 0.0),
                    "text": node.get_text()[:300] + "..."
                })
                
        return {
            "answer": str(response),
            "sources": sources
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)