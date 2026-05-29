import os
import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP, INGEST_BATCH
from embeddings import get_embedding_function
from db import get_collection

def load_pdf_pages(pdf_path):
    """
    pdf 페이지별 텍스트, 번호 추출
    """
    doc = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text()
        if text.strip():
            pages.append({"text": text, "page": i + 1})
    return pages

def ingest():
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    embed_fn = get_embedding_function()
    collection = get_collection()

    pdf_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".pdf")]
    print(f"PDF {len(pdf_files)}개 발견")

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

        for i in range(0, len(ids), INGEST_BATCH):
            collection.upsert(
                ids=ids[i:i+INGEST_BATCH],
                documents=documents[i:i+INGEST_BATCH],
                embeddings=embeddings[i:i+INGEST_BATCH],
                metadatas=metadatas[i:i+INGEST_BATCH],
            )
        print(f"{filename} 완료 — 청크 {len(ids)}개 저장")

if __name__ == "__main__":
    ingest()
