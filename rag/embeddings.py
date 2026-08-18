from dotenv import load_dotenv 
from openai import OpenAI

load_dotenv()

client = OpenAI()

def embed_texts(texts):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )

    return [item.embedding for item in response.data]