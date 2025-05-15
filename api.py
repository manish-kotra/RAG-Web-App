import os
import uvicorn
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from textRAG import textRAG

app = FastAPI(
    title="RAG API",
    description="API for Retrieval Augmented Generation using LangChain and Gemini",
    version="1.0.0",
)

@app.on_event("startup")
async def startup_db_client():
    app.rag_pipeline = textRAG(folder_path="pdfs")

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
    
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

