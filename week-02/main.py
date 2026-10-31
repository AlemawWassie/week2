"""Phone Repair FAQ Semantic Search — Week 02.
Run from this folder with: python task.py
"""
from pathlib import Path
import pandas as pd
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

PROJECT_FOLDER = Path(__file__).resolve().parent
CSV_FILE = PROJECT_FOLDER / "data" / "faqs.csv"
DB_FOLDER = PROJECT_FOLDER / "chroma_db"
COLLECTION_NAME = "phone_repair_faqs"


def load_faqs(file_path=CSV_FILE):
    """Load the CSV and check that required columns exist."""
    data = pd.read_csv(file_path)
    required = {"question", "answer", "category"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError("Missing CSV columns: " + ", ".join(sorted(missing)))
    return data


def clean_faqs(data):
    """Remove incomplete rows, extra spaces, and duplicate questions."""
    cleaned = data.copy()
    for column in ["question", "answer", "category"]:
        cleaned[column] = cleaned[column].astype("string").str.strip()
    cleaned = cleaned.dropna(subset=["question", "answer", "category"])
    cleaned = cleaned[
        (cleaned["question"] != "") &
        (cleaned["answer"] != "") &
        (cleaned["category"] != "")
    ]
    cleaned = cleaned.drop_duplicates(subset=["question"], keep="first")
    return cleaned.reset_index(drop=True)


def make_chunks(data):
    """Use one question-answer pair as one text chunk."""
    return [
        f"Question: {row['question']}\\nAnswer: {row['answer']}"
        for _, row in data.iterrows()
    ]


def save_faqs(data, client, embedding_function):
    """Store FAQ text and metadata in Chroma."""
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_function,
        metadata={"description": "Sample phone repair FAQs"},
    )
    chunks = make_chunks(data)
    ids = [f"faq-{i:03d}" for i in range(len(data))]
    metadata = [
        {"question": row["question"], "category": row["category"], "answer": row["answer"]}
        for _, row in data.iterrows()
    ]
    collection.upsert(ids=ids, documents=chunks, metadatas=metadata)
    return collection


def search_faqs(query, collection, limit=3):
    """Print up to three FAQs close in meaning to the user's query."""
    if not query.strip():
        print("Please enter a question.")
        return
    count = collection.count()
    if count == 0:
        print("The FAQ collection is empty.")
        return
    results = collection.query(
        query_texts=[query],
        n_results=min(limit, count),
        include=["documents", "metadatas", "distances"],
    )
    print(f"\\nYour question: {query}")
    print("=" * 58)
    for rank, (document, metadata, distance) in enumerate(
        zip(results["documents"][0], results["metadatas"][0], results["distances"][0]),
        start=1,
    ):
        print(f"{rank}. {metadata['question']} ({metadata['category']})")
        print(metadata["answer"])
        print(f"Distance: {distance:.4f}\\n")


def main():
    """Run the project from the terminal."""
    faqs = clean_faqs(load_faqs())
    print(f"FAQs ready after cleaning: {len(faqs)}")
    embedding_function = SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    client = chromadb.PersistentClient(path=str(DB_FOLDER))
    collection = save_faqs(faqs, client, embedding_function)
    print(f"FAQs stored in Chroma: {collection.count()}")
    query = input("\\nType a phone repair question (or press Enter to quit): ")
    if query.strip():
        search_faqs(query, collection)


if __name__ == "__main__":
    main()
