import chromadb
import time
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from config import LLM_MODEL, TOP_K, CHROMA_DIR
from embeddings import get_embedding_function

load_dotenv()

CHUNK_SIZES = [300, 500, 700]

TEST_QUESTIONS = [
    "RAG 시스템의 Retrieval 모듈에서 키워드 기반 검색과 임베딩 기반 검색의 차이점은?",
    "LLM의 할루시네이션을 완화하기 위해 프롬프트 제약 방식은 어떻게 작동하는가?",
    "공공서비스에 RAG 기술을 적용할 때 예상되는 장점과 문제점은?",
    "RAG 시스템에서 임베딩 모델의 역할은 무엇인가?",
    "RAG와 파인튜닝의 비용 비교 수치는?",
]

PROMPT_TEMPLATE = PromptTemplate.from_template(
    "당신은 논문 내용을 기반으로 질문에 답변하는 AI입니다.\n"
    "아래 컨텍스트만 사용하여 답변하고, 모르면 모른다고 하세요.\n\n"
    "[컨텍스트]\n{context}\n\n"
    "[질문]\n{question}"
)

def run_query(question, collection, embed_fn, llm):
    question_vector = embed_fn.embed_query(question)
    results = collection.query(
        query_embeddings=[question_vector],
        n_results=TOP_K,
        include=["documents", "metadatas"],
    )
    chunks = results["documents"][0]
    metas = results["metadatas"][0]

    context = "\n\n".join(
        f"[{m['source']}, p.{m['page']}]\n{doc}"
        for doc, m in zip(chunks, metas)
    )

    prompt = PROMPT_TEMPLATE.format(context=context, question=question)
    response = llm.invoke(prompt)

    sources = list({f"{m['source']} p.{m['page']}" for m in metas})
    return response.content, sources

if __name__ == "__main__":
    embed_fn = get_embedding_function()
    llm = ChatGoogleGenerativeAI(model=LLM_MODEL)
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    for chunk_size in CHUNK_SIZES:
        try:
            collection = client.get_or_create_collection(name=f"paper_chunks_{chunk_size}")
        except Exception:
            print(f"[건너뜀] paper_chunks_{chunk_size} 컬렉션 없음 — ingest 먼저 실행하세요")
            continue
        print(f"\n{'='*60}")
        print(f"청크 크기: {chunk_size}자")
        print(f"{'='*60}")

        for i, question in enumerate(TEST_QUESTIONS, 1):
            print(f"\n[Q{i}] {question}")
            answer, sources = run_query(question, collection, embed_fn, llm)
            #print(f"[답변] {answer[:300]}{'...' if len(answer) > 300 else ''}")
            print(f"[답변] {answer}")
            print(f"[출처] {', '.join(sources)}")
            print("-" * 40)
            time.sleep(4)
