import os
from pathlib import Path

from langchain_core.documents import Document
from langchain_docling import DoclingLoader
from langchain_docling.loader import ExportType
from docling.chunking import HybridChunker
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from transformers import AutoTokenizer, AutoModel
from dotenv import load_dotenv
# from ragatouille import RAGPretrainedModel
from langchain.retrievers import ContextualCompressionRetriever


class textRAG:
    def __init__(self, folder_path: str = "pdfs"):
        load_dotenv()
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
        os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
        self.folder_path = Path(folder_path)
        self.persist_directory = "./test_chroma_db"
        self.tokenizer = AutoTokenizer.from_pretrained("avsolatorio/GIST-large-Embedding-v0", trust_remote_code=True)
        self.embedding_model = HuggingFaceEmbeddings(model_name="avsolatorio/GIST-large-Embedding-v0")
        self.max_chunk_size = self.tokenizer.model_max_length 
        self.chunker = HybridChunker(
            tokenizer=self.tokenizer,
            max_chunk_size=self.max_chunk_size,
            merge_peers=True,)
        
        self.vector_store = Chroma(collection_name="collection",
                                    embedding_function=self.embedding_model,
                                    persist_directory=self.persist_directory,)
        
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)

        self.prompt = ChatPromptTemplate.from_template("""
                        You are an AI assistant that provides accurate information based on the given context.
                        
                        Context:
                        {context}
                        
                        Question:
                        {question}
                        
                        Provide a comprehensive answer based on the context. If the context doesn't contain the information needed to answer the question, say so clearly.
                        """)
        
        # # Initialize ColBERT model for reranking
        # self.colbert_model = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")


    def find_documents(self) -> list:
        documents = []
        for root, dirs, files in os.walk(self.folder_path):
            for file in files:
                if file.endswith(".pdf") or file.endswith(".txt"):
                    documents.append(Path(os.path.join(root, file)))
        return documents
    

    def index_documents(self, documents: list) -> None:
        docloader = DoclingLoader(file_path=documents,
                        export_type=ExportType.DOC_CHUNKS,
                        chunker=self.chunker)
        documents = docloader.load()

        processed_docs = []
        for doc in documents:
            orig_metadata = doc.metadata
            filename = Path(orig_metadata.get("source", "")).name
            source = str(orig_metadata.get("source", ""))
            processed_doc = Document(
                page_content=doc.page_content,
                metadata={
                    "source": source,
                    "filename": filename,
                    # "page": page_num
                    }
            )
            processed_docs.append(processed_doc)

        self.vector_store.add_documents(processed_docs)
        return processed_docs

    
    def load_from_db(self) -> list:
        loaded_docs = self.vector_store.get_all_documents()
        return loaded_docs
    
    
    def query_documents(self, query: str, use_reranker: bool = True) -> str:
        retriever = self.vector_store.as_retriever(
                        search_type="mmr", 
                        search_kwargs={"k": 10, "fetch_k": 20}
                    )
        
        # if use_reranker:
        #     compressor = self.colbert_model.as_langchain_document_compressor()
            
        #     compression_retriever = ContextualCompressionRetriever(
        #         base_compressor=compressor,
        #         base_retriever=retriever
        #     )
            
        #     reranked_docs = compression_retriever.invoke(query)
        #     context_str = "\n\n\n".join([doc.page_content for doc in reranked_docs])

        # else:
        retrieved_docs = retriever.invoke(query)
        context_str = "\n\n\n".join([doc.page_content for doc in retrieved_docs])
            
        return context_str
    

    def generate_response(self, query: str, context: str) -> str:
        rag_chain = (
              self.prompt
            | self.llm
            | StrOutputParser()
        )
        return rag_chain.invoke({"context": context, "question": query})
        

if __name__ == "__main__":
    load_dotenv()
    textRAG_instance = textRAG(folder_path="pdfs")
    
    # Check if we need to index new documents
    index_new_docs = input("Do you want to index new documents? (y/n): ").lower() == 'y'
    
    if index_new_docs:
        # Find and index new documents
        documents = textRAG_instance.find_documents()
        print("-" * 20)
        print(f"Found {len(documents)} documents in the directory.")
        for doc in documents:
            print(f"- {doc}")
        print("-" * 20, "\n")
        
        loaded_docs = textRAG_instance.index_documents(documents)
        print(f"Indexed {len(loaded_docs)} documents into the vector store.")
    
    while True:
        query = input("Enter your query: ")
        if query.lower() == "exit":
            break
        
        else:
            # Use ColBERT reranking
            context = textRAG_instance.query_documents(query, use_reranker=True)

            response = textRAG_instance.generate_response(query, context)
            print(response)