# RoboPaper: Research Intelligence

A specialized Retrieval-Augmented Generation (RAG) system for the intersection of **Robotics** and **Machine Learning**.

## Overview

This project was built to practice concepts from a DataCamp RAG course. It creates a local intelligence engine that ingests scientific abstracts from **arXiv**, indexes them using semantic vector embeddings, and uses a Large Language Model to synthesize evidence-based answers.

**Scope**: Strictly limited to ML applications in Robotics.

## System Architecture

The system operates in two distinct phases:

1.  **Ingestion (Static)**:
    - We pre-fetch **75 research papers** from arXiv matching the query `"robot reinforcement learning"`.
    - These 75 papers are fixed in our local database. The system does *not* go to the internet for every new question.
    
2.  **Retrieval (Dynamic)**:
    - When you ask a question, the system searches *within* those 75 pre-loaded papers.
    - It mathematically selects the top **10 specific text snippets** that best match your question.

```text
A. SETUP PHASE (Once)
   arXiv API -> Fetch 75 Papers -> ChromaDB (Local Index)

B. QUERY PHASE (Per Question)
   Your Question -> Cosine Search (Top 10 chunks) -> LLM -> Answer
```

## Tech Stack

- **Interface**: Streamlit (Minimalist UI)
- **Orchestration**: LangChain
- **Database**: ChromaDB
- **Model**: HuggingFace Inference API

## Setup

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Environment**:
   Create a `.env` file:
   ```env
   HUGGINGFACEHUB_API_TOKEN=hf_your_token_here
   ```

## Usage

1. **Load Data**: Fetch 75+ papers from arXiv.
   ```bash
   python vector_rag/scripts/00_load.py
   ```

2. **Index**: Embed and store the knowledge.
   ```bash
   python vector_rag/scripts/01_index.py
   ```

3. **Launch Interface**:
   ```bash
   streamlit run app.py
   ```

## Author
**Anas**
