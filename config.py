EMBEDDING_MODEL  = "jhgan/ko-sroberta-multitask"
EMBEDDING_DEVICE = "cpu"
#LLM_MODEL        = "gemini-2.5-flash"
LLM_MODEL        = "gemini-2.5-flash-lite"
#LLM_MODEL        = "gemini-3.1-flash-lite"

CHUNK_SIZE       = 500
CHUNK_OVERLAP    = 50

TOP_K            = 3
MAX_TOKENS       = 2048

CHROMA_DIR       = "./chroma_db"
COLLECTION_NAME  = "paper_chunks"
DATA_DIR         = "./data"
INGEST_BATCH     = 100