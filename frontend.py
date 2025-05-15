import streamlit as st
import requests
import json
from pathlib import Path
import os

API_URL = "http://localhost:8000"

# Set page configuration
st.set_page_config(
    page_title="RAG Document Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# App title and description
st.title("📚 RAG Document Assistant")

st.sidebar.title("Settings")

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
    

# Check API health
api_status = check_api_health()
if api_status:
    st.sidebar.success("API is running")
else:
    st.sidebar.error("API is not running. Please start the API server.")


# Display pdfs available in the directory
pdfs = loaded_docs()
if pdfs:
    st.sidebar.subheader("Loaded PDFs")
    pdf_list = pdfs.get("documents", [])
    if pdf_list:
        for pdf in pdf_list:
            st.sidebar.write(f"- {pdf}")
    else:
        st.sidebar.write("No PDFs loaded yet.")

# Q&A Section
st.subheader("Ask a Question")
question = st.text_input("Enter your question:")

if st.button("Submit"):
    if api_status and question:
        with st.spinner("Getting answer..."):
            result = query_api(question)
            if result:
                st.subheader("Answer")
                st.write(result["answer"])
                st.subheader("Source Documents")
                st.write(result["source_documents"])
    else:
        st.warning("Please enter a question.")




# response = query_api(question)