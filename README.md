# robopaper

Ask questions about robotics ML papers and get answers grounded in retrieved documents with citations.

Two implementations of the same idea: question-answering over robotics ML papers with citations.

## Structure
- vector_rag/ : classic RAG (Load → Split → Embed → Store → Retrieve → Generate)
- graph_rag/  : Graph RAG (graph construction + Neo4j + Cypher retrieval → Generate)

## Setup
1) Create venv (already done):
   python -m venv .venv
   source .venv/bin/activate

2) Install deps:
   pip install -r requirements.txt

3) Add API key:
   cp .env.example .env
   (edit .env and set OPENAI_API_KEY)

## Run (later)
Vector RAG:
  python vector_rag/scripts/00_load.py
  python vector_rag/scripts/01_index.py
  python vector_rag/scripts/02_ask.py "your question"

Graph RAG:
  docker compose up -d   (inside graph_rag when you add docker-compose.yml)
  python graph_rag/scripts/00_load.py
  python graph_rag/scripts/10_build_graph.py
  python graph_rag/scripts/12_ask.py "your question"
