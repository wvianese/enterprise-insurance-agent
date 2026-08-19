import json

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

def save_vector_store(vector_store, file_path="data/vector_store.json"): #Vector_store is the list of chunks. Each chunk contains the chunk_id, source, text, and embedding
    with open(file_path, "w") as temporary_file_name: #Open the file in write mode
        json.dump(vector_store, temporary_file_name, indent=2) #Convert the list of dictionaries into JSON format and write it into the file

        #"With" ensures the file is properly closed after writing

if __name__ == "__main__":
    vector_store = build_vector_store()
    save_vector_store(vector_store)

    #Only build and save embeddings when this file is run directly, not when it is imported / called by the LLM
    #Used when the policy documents are updated or new ones are added

def load_vector_store(file_path="data/vector_store.json"): #The function called by the LLM to load the vector store when it is needed for searching
    with open(file_path, "r") as temporary_file_name: #Open the file in read mode
        return json.load(temporary_file_name) #Convert the JSON data back into a list of dictionaries and return it

        #"With" ensures the file is properly closed after writing

