from rag.retriever import search_policies

def search_policy_documents(query, top_k=3):
    results = search_policies(query, top_k=top_k)

    retrieved_chunks = []

    for result in results:

        retrieved_chunks.append(
            {
                "chunk_id": result["chunk"]["chunk_id"],
                "source": result["chunk"]["source"],
                "text": result["chunk"]["text"],
                "score": float(result["score"])
            }
        ) #Eveything from the chunk but the embedding is returned to the LLM, as the embedding is only used for similarity scoring and is not needed for the LLM to generate a response

    return retrieved_chunks