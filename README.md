# RoboPaper: ArXiv Vector RAG

This project is a minimal implementation of Retrieval-Augmented Generation (RAG) using research papers from arXiv. I built this as a follow-up to a DataCamp RAG course to practicalize the concepts of document loading, embedding generation, and vector-based retrieval.

## Overview

The system fetches research abstracts from arXiv, indexes them into a local ChromaDB vector store using HuggingFace embeddings, and uses the HuggingFace Inference API to generate answers grounded in the retrieved context.

## Tech Stack

- LangChain (Orchestration)
- ChromaDB (Vector Store)
- HuggingFace (Local Embeddings & Inference API)
- Streamlit (Web Interface)
- arXiv API (Data Source)

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Configure environment:
   Create a `.env` file with your HuggingFace token:
   ```env
   HUGGINGFACEHUB_API_TOKEN=your_token_here
   ```

## Usage

1. Load papers:
   ```bash
   python vector_rag/scripts/00_load.py
   ```

2. Index documents:
   ```bash
   python vector_rag/scripts/01_index.py
   ```

3. Run interface:
   ```bash
   streamlit run app.py
   ```

Alternatively, use `python vector_rag/scripts/02_ask.py "Your question"` for CLI access.

## Project Structure

- app.py: Graphical user interface.
- vector_rag/scripts/: Core scripts for loading, indexing, and querying.
- chroma_db/: Local persistent database.
