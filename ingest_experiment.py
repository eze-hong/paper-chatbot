import os
import fitz
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import DATA_DIR, CHUNK_OVERLAP, CHROMA_DIR
from embeddings import get_embedding_function

CHUNK_SIZES = [300, 500, 700]

def load_pdf_pages(pdf_path):
    doc = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text()
        if text.strip():
            pages.append({"text": text, "page": i + 1})
    return pages

def ingest_for_chunk_size(chunk_size, embed_fn, client):
    collection_name = f"paper_chunks_{chunk_size}"
    collection = client.get_or_create_collection(name=collection_name)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=CHUNK_OVERLAP,
    )

    pdf_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".pdf")]
    total_chunks = 0

    for filename in pdf_files:
        pdf_path = os.path.join(DATA_DIR, filename)
        pages = load_pdf_pages(pdf_path)
        stem = os.path.splitext(filename)[0]

        ids, documents, embeddings, metadatas = [], [], [], []

        for page_data in pages:
            chunks = splitter.split_text(page_data["text"])
            for idx, chunk in enumerate(chunks):
                chunk_id = f"{stem}_p{page_data['page']}_c{idx}"
                ids.append(chunk_id)
                documents.append(chunk)
                embeddings.append(embed_fn.embed_query(chunk))
                metadatas.append({"source": filename, "page": page_data["page"]})

        for i in range(0, len(ids), 100):
            collection.upsert(
                ids=ids[i:i+100],
                documents=documents[i:i+100],
                embeddings=embeddings[i:i+100],
                metadatas=metadatas[i:i+100],
            )
        total_chunks += len(ids)

    print(f"[{chunk_size}자] 총 청크 수: {total_chunks}")
    return total_chunks

if __name__ == "__main__":
    embed_fn = get_embedding_function()
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    for chunk_size in CHUNK_SIZES:
        ingest_for_chunk_size(chunk_size, embed_fn, client)
