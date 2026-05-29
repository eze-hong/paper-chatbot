from langchain_huggingface import HuggingFaceEmbeddings
from config import EMBEDDING_MODEL, EMBEDDING_DEVICE

def get_embedding_function():
  return HuggingFaceEmbeddings(
      model_name=EMBEDDING_MODEL,
      model_kwargs={"device": EMBEDDING_DEVICE},
  )
