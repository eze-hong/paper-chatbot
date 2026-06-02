import time
import chromadb
import pandas as pd
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from datasets import Dataset

from config import LLM_MODEL, TOP_K, CHROMA_DIR
from embeddings import get_embedding_function
from ragas.metrics import faithfulness, answer_relevancy

answer_relevancy.strictness = 1

CHUNK_SIZES = [300, 500, 700]

TEST_QUESTIONS = [
    "RAG 시스템의 Retrieval 모듈에서 키워드 기반 검색과 임베딩 기반 검색의 차이점은?",
    "LLM의 할루시네이션을 완화하기 위해 프롬프트 제약 방식은 어떻게 작동하는가?",
    "공공서비스에 RAG 기술을 적용할 때 예상되는 장점과 문제점은?",
    "RAG 시스템에서 임베딩 모델의 역할은 무엇인가?",
    "RAG와 파인튜닝의 비용 비교 수치는?",
]

PROMPT_TEMPLATE = (
    "당신은 논문 내용을 기반으로 질문에 답변하는 AI입니다.\n"
    "아래 컨텍스트만 사용하여 답변하고, 모르면 모른다고 하세요.\n\n"
    "[컨텍스트]\n{context}\n\n"
    "[질문]\n{question}"
)


def collect_rows(chunk_size, client, embed_fn, llm) -> list[dict]:
    collection = client.get_or_create_collection(name=f"paper_chunks_{chunk_size}")
    rows = []

    for question in TEST_QUESTIONS:
        q_vec = embed_fn.embed_query(question)
        results = collection.query(
            query_embeddings=[q_vec],
            n_results=TOP_K,
            include=["documents"],
        )
        chunks = results["documents"][0]
        context = "\n\n".join(chunks)
        #answer = llm.invoke(PROMPT_TEMPLATE.format(context=context, question=question)).content
        raw = llm.invoke(PROMPT_TEMPLATE.format(context=context, question=question))
        answer = raw.content[0]['text'] if isinstance(raw.content, list) else raw.content
        
        rows.append({
            "question": question,
            "answer": answer,
            "contexts": chunks,
        })
        time.sleep(5)

    return rows


if __name__ == "__main__":
    load_dotenv()
    embed_fn = get_embedding_function()
    llm = ChatGoogleGenerativeAI(model=LLM_MODEL, n=1)
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    ragas_llm = LangchainLLMWrapper(llm)
    ragas_embeddings = LangchainEmbeddingsWrapper(embed_fn)

    first = True

    for chunk_size in CHUNK_SIZES:
        print(f"\n{'='*60}")
        print(f"청크 크기: {chunk_size}자")
        print(f"{'='*60}")

        rows = collect_rows(chunk_size, client, embed_fn, llm)

        dataset = Dataset.from_list(rows)
        result = evaluate(
            dataset=dataset,
            metrics=[faithfulness, answer_relevancy],
            llm=ragas_llm,
            embeddings=ragas_embeddings,
        )

        df = result.to_pandas()
        df.insert(0, "chunk_size", chunk_size)
        df.insert(1, "question", [r["question"] for r in rows])

        print(f"\n{'질문':<45} {'faithfulness':>14} {'answer_relevancy':>17}")
        print("-" * 78)
        for _, row in df.iterrows():
            q = row["question"][:43] + ".." if len(row["question"]) > 45 else row["question"]
            print(f"{q:<45} {row['faithfulness']:>14.2f} {row['answer_relevancy']:>17.2f}")
        print("-" * 60)
        print(f"평균 Faithfulness: {df['faithfulness'].mean():.2f} | 평균 Answer Relevancy: {df['answer_relevancy'].mean():.2f}")

        df[["chunk_size", "question", "faithfulness", "answer_relevancy"]].to_csv(
            "ragas_results.csv",
            mode="w" if first else "a",
            header=first,
            index=False,
            encoding="utf-8-sig",
        )
        first = False

        if chunk_size != CHUNK_SIZES[-1]:
            time.sleep(5)

    print("\n\n완료 — ragas_results.csv 저장됨")
