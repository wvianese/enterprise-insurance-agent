import numpy as np

from rag.embeddings import embed_texts
from rag.vector_store import load_vector_store

def cosine_similarity(vector_a, vector_b):
    return np.dot(vector_a, vector_b) / (np.linalg.norm(vector_a) * np.linalg.norm(vector_b))
    #Compare two vectors by calculating the cosine of the angle between them
    #Cosine similarity ranges from -1 to 1, where 1 means the vectors point in the same direction, 0 means they are orthogonal, and -1 means they point in opposite directions
    
def search_policies(query, top_k=3):
    store = load_vector_store() #Load the pre-built vector store
    query_embedding = embed_texts([query])[0] #Create an embedding for the query

    scored_chunks = []

    for chunk in store:
        score = cosine_similarity(query_embedding, chunk["embedding"]) #Calculate the similarity score between the query and each chunk
        scored_chunks.append(
            {
                "chunk": chunk, #Contains the chunk_id, source, text, and embedding of the chunk
                "score": score
            }
        ) #Store the chunk and its similarity score in a list

    scored_chunks.sort(key=lambda item: item["score"], reverse=True) #Sort the chunks by their similarity score in descending order

    return scored_chunks[:top_k] #Return the top_k most similar chunks 