"""
app/services/rag.py
Looks up the style guide for whatever clothing type the CNN detects, plus
searches for extra relevant info based on the user's question.
"""

import pathlib
import chromadb
from chromadb.utils import embedding_functions

KB_DIR = pathlib.Path(__file__).resolve().parent.parent.parent / "data" / "knowledge_base"
CHROMA_PATH = pathlib.Path(__file__).resolve().parent.parent.parent / "data" / "chroma_store"

_embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)


def _load_docs() -> dict[str, str]:
    docs = {}
    for path in KB_DIR.glob("*.md"):
        docs[path.stem] = path.read_text()
    return docs


def build_vector_store() -> chromadb.Collection:
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    try:
        client.delete_collection("fashion_kb")
    except Exception:
        pass
    collection = client.create_collection(name="fashion_kb", embedding_function=_embedding_fn)

    docs = _load_docs()
    collection.add(
        ids=list(docs.keys()),
        documents=list(docs.values()),
        metadatas=[{"class_name": k} for k in docs.keys()],
    )
    print(f"Vector store built with {len(docs)} documents.")
    return collection


def get_collection() -> chromadb.Collection:
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    return client.get_collection("fashion_kb", embedding_function=_embedding_fn)


def retrieve_context(predicted_class: str, user_question: str, n_results: int = 2) -> str:
    collection = get_collection()

    primary = collection.get(ids=[predicted_class])
    primary_text = primary["documents"][0] if primary["documents"] else ""

    semantic_results = collection.query(query_texts=[user_question], n_results=n_results)
    semantic_docs = semantic_results["documents"][0] if semantic_results["documents"] else []
    semantic_ids = semantic_results["ids"][0] if semantic_results["ids"] else []

    combined = {predicted_class: primary_text}
    for doc_id, doc_text in zip(semantic_ids, semantic_docs):
        if doc_id not in combined:
            combined[doc_id] = doc_text

    return "\n\n---\n\n".join(combined.values())


if __name__ == "__main__":
    build_vector_store()
