"""
Vector RAG - Step 3: Ask Questions
====================================
Retrieves relevant context from ChromaDB and generates answers using HuggingFace API.

Usage:
    python 02_ask.py "Your question here"
    
Prerequisites:
    - Run 00_load.py and 01_index.py first
    - Set HUGGINGFACEHUB_API_TOKEN in .env
"""
import os
import sys
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from huggingface_hub import InferenceClient

# Load environment variables
load_dotenv()


def format_docs(docs: list) -> str:
    """Format retrieved documents into a single context string."""
    return "\n\n".join(doc.page_content for doc in docs)


def ask_question(question: str, persist_dir: str = "./chroma_db") -> str:
    """
    Retrieve relevant context and generate an answer.
    
    Args:
        question: The user's question
        persist_dir: Path to ChromaDB vector store
        
    Returns:
        Generated answer string
    """
    # Step 1: Load vector store
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings
    )
    
    # Step 2: Retrieve relevant documents
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    docs = retriever.invoke(question)
    context = format_docs(docs)
    
    if not context.strip():
        return "No relevant documents found."
    
    # Step 3: Build prompt with context
    prompt = f"""Use the following context to answer the question. 
If you don't know the answer, say "I don't know" don't make things up.

Context:
{context}

Question: {question}

Answer:"""
    
    # Step 4: Call HuggingFace Inference API
    hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not hf_token:
        return "ERROR: HUGGINGFACEHUB_API_TOKEN not set in .env"
    
    client = InferenceClient(token=hf_token)
    
    response = client.chat_completion(
        messages=[{"role": "user", "content": prompt}],
        model="Qwen/Qwen2.5-72B-Instruct",
        max_tokens=256,
    )
    
    return response.choices[0].message.content


if __name__ == "__main__":
    # Get question from command line or use default
    if len(sys.argv) > 1:
        question = sys.argv[1]
    else:
        question = "How is machine learning used in robotics?"
    
    print(f"Question: {question}")
    print("Thinking...")
    
    answer = ask_question(question)
    print(f"Answer: {answer}")
