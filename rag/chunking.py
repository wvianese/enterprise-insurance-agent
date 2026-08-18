from pathlib import Path

POLICY_DIR = Path("data/policies")

def load_policy_documents():
    documents = []

    for file_path in POLICY_DIR.glob("*.txt"):
        text = file_path.read_text()

        documents.append(
            {
                "source": file_path.name,
                "text": text
            }
        )

    return documents

def chunk_documents(documents, chunk_size=500, overlap=100):
    chunks = []

    for document in documents:
        text = document["text"]

        chunk_index = 0
        start = 0

        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end]

            chunks.append(
                {
                    "chunk_id": f"{document['source']}_chunk_{chunk_index}",
                    "source": document["source"],
                    "text": chunk_text
                }
            )

            chunk_index += 1
            start += chunk_size - overlap

    return chunks