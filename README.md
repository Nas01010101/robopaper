# RoboPaper: Research Intelligence

A specialized Retrieval-Augmented Generation (RAG) system for the intersection of **Robotics** and **Machine Learning**.

## Overview

This project was built to practicalize concepts from a DataCamp RAG course. It creates a local intelligence engine that ingests scientific abstracts from **arXiv**, indexes them using semantic vector embeddings, and uses a Large Language Model to synthesize evidence-based answers.

**Scope**: Strictly limited to ML applications in Robotics.

## System Architecture

```text
SOURCE: arXiv API (Real-time Abstracts)
   ↓
VECTOR STORE: ChromaDB (Local Persistence)
   ↓
RETRIEVAL: HuggingFace Embeddings (all-MiniLM-L6-v2)
   ↓
GENERATION: Qwen-2.5-72B-Instruct (via Inference API)
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
