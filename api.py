import os
import uvicorn
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException, UploadFile, File, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from textRAG import textRAG
import tempfile
import shutil
from pathlib import Path

app = FastAPI(
    title="RAG API",
    description="API for Retrieval Augmented Generation using LangChain and Gemini",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_db_client():
    app.rag_pipeline = textRAG(folder_path="pdfs", resume_file="ManishKumarResume.pdf")

# Query model
class QueryRequest(BaseModel):
    query: str
    # collection_name: Optional[str] = "documents"
    # temperature: Optional[float] = 0.2
    # max_tokens: Optional[int] = 1024

class DocumentUploadRequest(BaseModel):
    pdf_directory: str = "pdfs"
    
# Response models
class QueryResponse(BaseModel):
    answer: str
    source_documents: Optional[str] = None

class DocumentsResponse(BaseModel):
    message: str
    document_count: int

class UploadResponse(BaseModel):
    message: str
    filename: str
    success: bool

class ListResponse(BaseModel):
    documents: list

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    try:
        context = app.rag_pipeline.query_documents(request.query)
        answer = app.rag_pipeline.generate_response(request.query, context)
        print(f"Answer: {answer}")
        return QueryResponse(answer=answer, source_documents=context)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.get("/loadedpdfs", response_model=ListResponse)
async def loaded_pdfs():
    try:
        documents = app.rag_pipeline.find_documents()
        return ListResponse(documents=documents)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    try:
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")
        
        # Read file content
        file_content = await file.read()
        
        # Add to RAG pipeline as temporary document
        temp_path = app.rag_pipeline.add_temporary_document(file_content, file.filename)
        
        return UploadResponse(
            message=f"File {file.filename} uploaded and indexed successfully",
            filename=file.filename,
            success=True
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/cleanup-session")
async def cleanup_session():
    try:
        app.rag_pipeline.cleanup_session_documents()
        return {"message": "Session documents cleaned up successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

