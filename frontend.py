import streamlit as st
import requests
import json
from pathlib import Path
import os
import atexit
import time

API_URL = "http://localhost:8000"

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
if 'api_status' not in st.session_state:
    st.session_state.api_status = None
if 'api_last_checked' not in st.session_state:
    st.session_state.api_last_checked = 0
if 'loaded_docs_cache' not in st.session_state:
    st.session_state.loaded_docs_cache = None

def cleanup_session():
    """Clean up session documents on app close"""
    try:
        response = requests.post(f"{API_URL}/cleanup-session")
        return response.status_code == 200
    except:
        return False

# App title and description
st.title("📚 RAG Document Assistant")
st.markdown("Ask questions about documents! ManishKumar's resume is always available, and you can upload additional PDFs for your session.")
st.markdown("*Disclaimer: This is a demo application with limited functionality. There is no chat history implemented due to memeory constraints, hence each question is independent of previous ones.  Please note that uploading a 4-page PDF may take 1–2 minutes to process.*")


st.sidebar.title("Settings")

# Cleanup on session end
if not st.session_state.session_cleaned:
    atexit.register(cleanup_session)

def check_api_health():
    try:
        response = requests.get(f"{API_URL}/health")
        return response.status_code == 200
    except:
        return False
    
def query_api(question):
    try:
        response = requests.post(f"{API_URL}/query", json={"query": question})
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"Exception occurred: {str(e)}")
        return None

def upload_file(file_bytes, filename):
    try:
        files = {'file': (filename, file_bytes, 'application/pdf')}
        response = requests.post(f"{API_URL}/upload", files=files)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error uploading file: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"Exception occurred during upload: {str(e)}")
        return None

def loaded_docs():
    try:
        response = requests.get(f"{API_URL}/loadedpdfs")
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"Exception occurred: {str(e)}")
        return None
    

# Check API health only on first load (session start)
if st.session_state.api_status is None:
    st.session_state.api_status = check_api_health()
    st.session_state.api_last_checked = time.time()

# Use cached API status
if st.session_state.api_status:
    st.sidebar.success("API is running")
else:
    st.sidebar.error("API is not running. Please start the API server.")

# Add refresh button for API status
if st.sidebar.button("Refresh APIs"):
    st.session_state.api_status = check_api_health()
    st.session_state.api_last_checked = time.time()
    st.session_state.loaded_docs_cache = None  # Clear docs cache too
    if st.session_state.api_status:
        st.session_state.loaded_docs_cache = loaded_docs()


# Display pdfs available in the directory - use cached data
if st.session_state.api_status and st.session_state.loaded_docs_cache is None:
    st.session_state.loaded_docs_cache = loaded_docs()

pdfs = st.session_state.loaded_docs_cache
if pdfs:
    st.sidebar.subheader("Available Documents")
    pdf_list = pdfs.get("documents", [])
    if pdf_list:
        st.sidebar.write("**Permanent:**")
        st.sidebar.write("- ManishKumarResume.pdf")
        
        if st.session_state.uploaded_files:
            st.sidebar.write("**Session (Temporary):**")
            for uploaded_file in st.session_state.uploaded_files:
                st.sidebar.write(f"- {uploaded_file}")
    else:
        st.sidebar.write("No PDFs loaded yet.")

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
            file_bytes = uploaded_file.read()
            result = upload_file(file_bytes, uploaded_file.name)
            
            if result and result.get('success'):
                st.session_state.uploaded_files.append(uploaded_file.name)
                st.sidebar.success(f"{uploaded_file.name} uploaded successfully!")
            else:
                st.sidebar.error(f"Failed to upload {uploaded_file.name}")

# Session cleanup button
if st.session_state.uploaded_files:
    if st.sidebar.button("Clear Session Documents"):
        with st.spinner("Cleaning up session documents..."):
            if cleanup_session():
                st.session_state.uploaded_files = []
                st.session_state.session_cleaned = True
                st.sidebar.success("Session documents cleared!")
                # st.experimental_rerun()
            else:
                st.sidebar.error("Failed to clear session documents")

# Q&A Section
st.subheader("💬 Ask a Question")
st.write("Ask questions about ManishKumar's resume or any uploaded documents.")

question = st.text_input("Enter your question:", placeholder="e.g., What is ManishKumar's experience in NLP?")

col1, col2 = st.columns([1, 4])
with col1:
    submit_button = st.button("Submit", type="primary")
with col2:
    if st.session_state.uploaded_files:
        st.info(f"{len(st.session_state.uploaded_files)} additional document(s) in this session")

if submit_button:
    if question and st.session_state.api_status:
        with st.spinner("Thinking..."):
            result = query_api(question)
            if result:
                st.subheader("Answer")
                st.write(result["answer"])
                
                with st.expander("Source Context"):
                    st.text(result["source_documents"])
    else:
        if not st.session_state.api_status:
            st.error("API is not running. Please start the API server.")
        elif not question:
            st.warning("Please enter a question.")
