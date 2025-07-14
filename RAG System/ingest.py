
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_astradb import AstraDBVectorStore
from dotenv import load_dotenv
import os
from data_converter import convert_data  

def ingest_data(status=None):
    load_dotenv()

    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    ASTRA_API_KEY = os.getenv("ASTRA_API_KEY")
    DB_ENDPOINT = os.getenv("ASTRA_ENDPOINT")
    DB_NAME = os.getenv("DB_NAME")

    if not all([GOOGLE_API_KEY, ASTRA_API_KEY, DB_ENDPOINT, DB_NAME]):
        raise ValueError("One or more environment variables are missing.")

    print("Initializing vector store...")
    gemini_embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    vector_store = AstraDBVectorStore(
        embedding=gemini_embedding,
        api_endpoint=DB_ENDPOINT,
        namespace="default_keyspace",
        token=ASTRA_API_KEY,
        collection_name="travel",
    )
    print("Vector store initialized.")
    if status is None:
        text_chunks = convert_data()  
        inserted_ids = vector_store.add_documents(text_chunks)
        return vector_store, inserted_ids
    else:
        return vector_store, []

if __name__ == "__main__":
    vector_store, inserted_ids = ingest_data()
    print("DB has been initialized.")
    if inserted_ids:
        print(f"Inserted {len(inserted_ids)} documents.")
