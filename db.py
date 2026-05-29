import chromadb
from config import CHROMA_DIR, COLLECTION_NAME

def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)
    return collection