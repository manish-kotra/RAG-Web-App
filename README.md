# RAG Web Application

A Retrieval Augmented Generation (RAG) application that uses LangChain, Chroma vector database, and Google's Gemini model to provide accurate responses based on your document collection.

## Overview

This application enables you to:
- Index PDF and text documents into a vector database
- Query your documents using natural language
- Get AI-generated responses based on the content of your documents
- Access the functionality through both an API and a web interface

## Architecture

The application consists of three main components:

1. **textRAG.py**: Core RAG implementation using LangChain
   - Document processing and chunking
   - Vector embeddings generation
   - Vector database management
   - Retrieval and generation pipeline

2. **api.py**: FastAPI-based REST API
   - Query endpoint
   - Document upload functionality
   - Health check endpoint

3. **frontend.py**: Web interface (optional)
   - User-friendly interface for querying documents

## Installation

### Prerequisites

- Python 3.8+
- pip (Python package installer)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/manish-kotra/RAG-Web-App.git
   cd RAG-Web-App
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file with your API keys:
   ```
   GOOGLE_API_KEY=your_google_api_key
   LANGSMITH_API_KEY=your_langsmith_api_key
   ```

## Usage

### Preparing Your Documents

1. Place your PDF or text documents in the `pdfs/` directory.

### Using the Command-line Interface

Run the RAG application directly:

```bash
python textRAG.py
```

You'll be prompted to index new documents or use the existing vector store. Then you can query your documents interactively.

### Using the API

Start the API server:

```bash
python api.py
```

The API will be available at http://localhost:8000

#### API Endpoints:

- `POST /query` - Query your documents
  ```json
  {
    "query": "What does the document say about..."
  }
  ```

- `GET /loadedpdfs` - List loaded documents
- `GET /health` - Health check endpoint

### API Documentation

When running, API documentation is available at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Technical Details

### Vector Store

The application uses Chroma DB for the vector database, stored in the `test_chroma_db/` directory.

### Embedding Model

Documents are embedded using the "GIST-large-Embedding-v0" model from Hugging Face.

### LLM

The application uses Google's Gemini model for response generation.

### Chunking Strategy

Documents are processed using a hybrid chunking strategy that balances context retention and chunk size.

## Development

### Project Structure

```
├── api.py                 # FastAPI implementation
├── textRAG.py             # Core RAG implementation
├── frontend.py            # Web interface
├── requirements.txt       # Python dependencies
├── pdfs/                  # Directory for document storage
└── test_chroma_db/        # Vector database storage
```

## License

[Specify your license here]

## Contributing

[Guidelines for contributing to the project]
