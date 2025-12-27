"""
Vector RAG - Step 1: Load Papers
================================
Fetches research papers from arXiv and saves them for indexing.

Usage:
    python 00_load.py
    
Output:
    papers.pkl - Pickled list of LangChain Document objects
"""
import arxiv
from langchain_core.documents import Document


def load_arxiv_papers(query: str, max_results: int = 20) -> list[Document]:
    """
    Load papers from arXiv based on a search query.
    
    Args:
        query: Search query (e.g., "robot reinforcement learning")
        max_results: Maximum number of papers to fetch
        
    Returns:
        List of LangChain Document objects with paper abstracts and metadata
    """
    print(f"Loading {max_results} papers for query: '{query}'...")
    
    # Initialize arXiv client and search
    client = arxiv.Client()
    search = arxiv.Search(query=query, max_results=max_results)
    
    # Convert results to LangChain Documents
    docs = []
    for result in client.results(search):
        doc = Document(
            page_content=result.summary,  # Paper abstract
            metadata={
                "Title": result.title,
                "Authors": ", ".join(a.name for a in result.authors),
                "Published": str(result.published),
                "URL": result.entry_id,
            }
        )
        docs.append(doc)
        print(f"  ✓ {result.title[:60]}...")
    
    print(f"Loaded {len(docs)} papers.")
    return docs


if __name__ == "__main__":
    import pickle
    
    # Fetch papers from arXiv
    papers = load_arxiv_papers(
        query="robot reinforcement learning",
        max_results=20
    )
    
    # Save for indexing
    with open("./papers.pkl", "wb") as f:
        pickle.dump(papers, f)
    print(f"Saved {len(papers)} papers to papers.pkl")
