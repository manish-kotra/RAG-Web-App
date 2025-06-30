import streamlit as st
import os
import tempfile
import shutil
from pathlib import Path
import time
from textRAG import textRAG

# Set page configuration
st.set_page_config(
    page_title="RAG Document Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize session state for tracking uploads
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []
if 'session_cleaned' not in st.session_state:
    st.session_state.session_cleaned = False
if 'rag_pipeline' not in st.session_state:
    st.session_state.rag_pipeline = None
if 'rag_initialized' not in st.session_state:
    st.session_state.rag_initialized = False

@st.cache_resource
def initialize_rag():
    """Initialize RAG pipeline with resume - cached for performance"""
    try:
        return textRAG(folder_path="pdfs", resume_file="ManishKumarResume.pdf")
    except Exception as e:
        st.error(f"Error initializing RAG system: {str(e)}")
        return None

def cleanup_session_documents():
    """Clean up session documents"""
    if st.session_state.rag_pipeline:
        try:
            st.session_state.rag_pipeline.cleanup_session_documents()
            return True
        except Exception as e:
            st.error(f"Error cleaning up session: {str(e)}")
            return False
    return False

def add_temporary_document(file_content, filename):
    """Add a temporary document to the RAG system"""
    if st.session_state.rag_pipeline:
        try:
            temp_path = st.session_state.rag_pipeline.add_temporary_document(file_content, filename)
            return temp_path is not None
        except Exception as e:
            st.error(f"Error adding document: {str(e)}")
            return False
    return False

def query_documents(question):
    """Query the RAG system"""
    if st.session_state.rag_pipeline:
        try:
            context = st.session_state.rag_pipeline.query_documents(question)
            answer = st.session_state.rag_pipeline.generate_response(question, context)
            return {"answer": answer, "source_documents": context}
        except Exception as e:
            st.error(f"Error querying documents: {str(e)}")
            return None
    return None

# App title and description
st.title("📚 RAG Document Assistant")
st.markdown("Ask questions about documents! ManishKumar's resume is always available, and you can upload additional PDFs for your session.")
st.markdown("*Disclaimer: This is a demo application with limited functionality. There is no chat history implemented due to memory constraints, hence each question is independent of previous ones. Please note that uploading a 4-page PDF may take 1–2 minutes to process.*")

# Initialize RAG system
if not st.session_state.rag_initialized:
    with st.spinner("Initializing RAG system... This may take a moment."):
        st.session_state.rag_pipeline = initialize_rag()
        if st.session_state.rag_pipeline:
            st.session_state.rag_initialized = True
            st.sidebar.success("RAG System Ready")
        else:
            st.sidebar.error("Failed to initialize RAG system")
            st.error("Failed to initialize the RAG system. Please check your environment variables and try again.")
            st.stop()

st.sidebar.title("Settings")

# Show system status
if st.session_state.rag_initialized:
    st.sidebar.success("RAG System Ready")
else:
    st.sidebar.error("RAG System Not Ready")

# Display available documents
st.sidebar.subheader("Available Documents")
st.sidebar.write("**Permanent:**")
st.sidebar.write("- ManishKumarResume.pdf")

if st.session_state.uploaded_files:
    st.sidebar.write("**Session (Temporary):**")
    for uploaded_file in st.session_state.uploaded_files:
        st.sidebar.write(f"- {uploaded_file} ")

# File Upload Section
st.sidebar.subheader("Upload Document")
uploaded_file = st.sidebar.file_uploader(
    "Upload a PDF for this session",
    type="pdf",
    help="Upload a PDF file to ask questions about. This will be removed when the session ends."
)

if uploaded_file is not None:
    if uploaded_file.name not in st.session_state.uploaded_files:
        with st.spinner(f"Uploading and processing {uploaded_file.name}..."):
            try:
                file_bytes = uploaded_file.read()
                success = add_temporary_document(file_bytes, uploaded_file.name)
                
                if success:
                    st.session_state.uploaded_files.append(uploaded_file.name)
                    st.sidebar.success(f"{uploaded_file.name} uploaded successfully!")
                else:
                    st.sidebar.error(f"Failed to upload {uploaded_file.name}")
            except Exception as e:
                st.sidebar.error(f"Error uploading file: {str(e)}")

# Session cleanup button
if st.session_state.uploaded_files:
    if st.sidebar.button("🗑️ Clear Session Documents"):
        with st.spinner("Cleaning up session documents..."):
            if cleanup_session_documents():
                st.session_state.uploaded_files = []
                st.session_state.session_cleaned = True
                st.sidebar.success("Session documents cleared!")
                st.rerun()
            else:
                st.sidebar.error("Failed to clear session documents")

# Q&A Section
st.subheader("💬 Ask a Question")
st.write("Ask questions about ManishKumar's resume or any uploaded documents.")

question = st.text_input("Enter your question:", placeholder="e.g., What is ManishKumar's experience in Python?")

col1, col2 = st.columns([1, 4])
with col1:
    submit_button = st.button("Submit", type="primary")
with col2:
    if st.session_state.uploaded_files:
        st.info(f"{len(st.session_state.uploaded_files)} additional document(s) in this session")

if submit_button:
    if question and st.session_state.rag_initialized:
        with st.spinner("Thinking..."):
            result = query_documents(question)
            if result:
                st.subheader("Answer")
                st.write(result["answer"])
                
                with st.expander("Source Context"):
                    st.text(result["source_documents"])
            else:
                st.error("Failed to get an answer. Please try again.")
    else:
        if not st.session_state.rag_initialized:
            st.error("RAG system is not ready. Please wait for initialization.")
        elif not question:
            st.warning("Please enter a question.")
