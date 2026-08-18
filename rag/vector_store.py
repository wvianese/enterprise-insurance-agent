from rag.chunking import load_policy_documents, chunk_documents
from rag.embeddings import embed_texts

def build_vector_store():
    documents = load_policy_documents() #Create list of all documents in policies
    chunks = chunk_documents(documents) #Chunks each document into smaller pieces

    texts = [chunk["text"] for chunk in chunks] #Strip away chunk_id and source, leaving only the text
    embeddings = embed_texts(texts) #Create embeddings for each chunk of text thus capturing its semantic meaning

    for chunk, embedding in zip(chunks, embeddings):
        chunk["embedding"] = embedding #Add the embedding to the chunk dictionary

    return chunks