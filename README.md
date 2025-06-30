# RAG Document Assistant 📚

A Retrieval Augmented Generation (RAG) application that allows you to ask questions about documents. The system includes ManishKumar's resume as a permanent document and supports uploading additional PDFs for temporary use during your session.

## Features

- **Permanent Resume**: ManishKumar's resume is always available in the vector database
- **Temporary Upload**: Upload additional PDFs for your current session
- **Auto Cleanup**: Session documents are automatically removed when the session ends
- **Smart Q&A**: Ask questions about any loaded document using Google Gemini
- **Web Interface**: User-friendly Streamlit frontend
- **API Backend**: FastAPI backend for document processing

## Architecture

- **Backend**: FastAPI with ChromaDB for vector storage
- **Frontend**: Streamlit web interface
- **AI Model**: Google Gemini 2.0 Flash for text generation
- **Embeddings**: HuggingFace GIST-large-Embedding-v0
- **Document Processing**: Docling for PDF processing and chunking

## Setup

### Prerequisites

- Python 3.8+
- Google API Key (for Gemini)

### Installation

1. **Clone and navigate to the project**:
   ```bash
   cd d:\Project\RAG
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   Create a `.env` file in the project root:
   ```env
   GOOGLE_API_KEY=your_google_api_key_here
   LANGSMITH_API_KEY=your_langsmith_api_key_here  # Optional
   ```

4. **Ensure the resume file exists**:
   Make sure `ManishKumarResume.pdf` is in the `pdfs/` folder

## Running the Application

### Option 1: Using the startup script (Recommended)
```bash
python run_app.py
```

### Option 2: Using the batch file (Windows)
```cmd
start.bat
```

### Option 3: Manual startup

1. **Start the API server**:
   ```bash
   uvicorn api:app --host 0.0.0.0 --port 8000 --reload
   ```

2. **Start the Streamlit frontend** (in another terminal):
   ```bash
   streamlit run frontend.py
   ```

## Usage

1. **Access the application**:
   - Frontend: http://localhost:8501
   - API Documentation: http://localhost:8000/docs

2. **Ask questions about the resume**:
   - The system always has access to ManishKumar's resume
   - Ask questions like "What is ManishKumar's experience in Python?"

3. **Upload additional documents**:
   - Use the sidebar to upload PDF files
   - These files are processed and added to the current session
   - Ask questions about the uploaded content

4. **Session management**:
   - Uploaded documents are automatically removed when the session ends
   - Use the "Clear Session Documents" button to manually clean up
   - The resume file remains permanently available

## API Endpoints

- `GET /health` - Health check
- `POST /query` - Ask questions about documents
- `POST /upload` - Upload a PDF for the current session
- `POST /cleanup-session` - Clean up session documents
- `GET /loadedpdfs` - List available documents

## File Structure

```
d:\Project\RAG\
├── api.py              # FastAPI backend
├── frontend.py         # Streamlit frontend
├── textRAG.py         # Core RAG implementation
├── run_app.py         # Startup script
├── start.bat          # Windows batch file
├── requirements.txt   # Python dependencies
├── README.md          # Documentation
├── .env              # Environment variables (create this)
├── pdfs/             # PDF storage
│   └── ManishKumarResume.pdf
└── test_chroma_db/   # Vector database storage
```

## Troubleshooting

1. **API not starting**: Check if port 8000 is available
2. **Streamlit not starting**: Check if port 8501 is available
3. **Missing resume**: Ensure `ManishKumarResume.pdf` is in the `pdfs/` folder
4. **Upload failures**: Check file size and ensure it's a valid PDF
5. **Google API errors**: Verify your `GOOGLE_API_KEY` in the `.env` file

## Development

To extend the application:

1. **Add new document types**: Modify the file upload validation in `api.py`
2. **Change embedding models**: Update the model in `textRAG.py`
3. **Customize UI**: Modify the Streamlit interface in `frontend.py`
4. **Add new endpoints**: Extend the FastAPI routes in `api.py`

## Dependencies

Key dependencies include:
- `fastapi` - Web API framework
- `streamlit` - Web interface
- `langchain` - LLM framework
- `chromadb` - Vector database
- `transformers` - HuggingFace models
- `docling` - Document processing
- `google-generativeai` - Google Gemini integration

See `requirements.txt` for the complete list.
