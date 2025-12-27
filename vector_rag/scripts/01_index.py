"""
Vector RAG - Step 2: Index Documents
=====================================
Splits documents into chunks, generates embeddings, and stores in ChromaDB.

Usage:
    python 01_index.py
    
Prerequisites:
    - Run 00_load.py first to create papers.pkl
    
Output:
    ./chroma_db/ - Persistent vector store
"""
import os
import pickle
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# Load environment variables
load_dotenv()


def index_documents(docs: list[Document], persist_dir: str = "./chroma_db") -> None:
    """
    Split documents into chunks and index them into ChromaDB.
    
    Args:
        docs: List of LangChain Document objects to index
        persist_dir: Directory to save the vector store
    """
    # Step 1: Split documents into chunks
    print("Splitting documents into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,     # Characters per chunk
        chunk_overlap=200    # Overlap between chunks
    )
    chunks = splitter.split_documents(docs)
    print(f"  Created {len(chunks)} chunks from {len(docs)} documents.")
    
    # Step 2: Initialize embeddings (runs locally, so it'sfree)
    print("Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # Step 3: Create and persist vector store
    print("Creating vector store...")
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir
    )
    print(f"  ✓ Indexed {len(chunks)} chunks to {persist_dir}")


if __name__ == "__main__":
    papers_file = "./papers.pkl"
    
    if os.path.exists(papers_file):
        # Load papers from 00_load.py
        with open(papers_file, "rb") as f:
            docs = pickle.load(f)
        print(f"Indexing {len(docs)} papers from {papers_file}...")
        index_documents(docs)
    else:
        print("No papers.pkl found. Run 00_load.py first!")
