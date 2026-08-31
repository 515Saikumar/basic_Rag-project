from src.data_loader import load_all_documents
from src.vectorstore import FaissVectorStore

from src.search import RAGSearch
from pathlib import Path

# Example usage
if __name__ == "__main__":

    base_dir = Path(__file__).resolve().parent
    docs = load_all_documents(base_dir / "src/data")
    store = FaissVectorStore("faiss_store")
    store.build_from_documents(docs)
    store.load()

    rag_search = RAGSearch()
    query = "Define trees?"
    summary = rag_search.search_and_summarize(query, top_k=3)
    print("Summary:", summary)
