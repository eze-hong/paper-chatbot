import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from config import LLM_MODEL, TOP_K
from embeddings import get_embedding_function
from db import get_collection

load_dotenv()

def query(question: str):
    embed_fn = get_embedding_function()
    collection = get_collection()

    question_vector = embed_fn.embed_query(question)
    results = collection.query(
        query_embeddings=[question_vector],
        n_results=TOP_K,
        include=["documents", "metadatas"],
    )

    chunks = results["documents"][0]
    metas = results["metadatas"][0]

    context = "\n\n".join(
        f"[출처: {m['source']}, p.{m['page']}]\n{doc}"
        for doc, m in zip(chunks, metas)
    )

    prompt_template = PromptTemplate.from_template(
        "당신은 논문 내용을 기반으로 질문에 답변하는 AI입니다.\n"
        "아래 컨텍스트만 사용하여 답변하고, 모르면 모른다고 하세요.\n\n"
        "[컨텍스트]\n{context}\n\n"
        "[질문]\n{question}"
    )

    llm = ChatGoogleGenerativeAI(model=LLM_MODEL)
    prompt = prompt_template.format(context=context, question=question)
    response = llm.invoke(prompt)

    print("\n[답변]")
    print(response.content)
    print("\n[참고 출처]")
    for m in metas:
        print(f"  - {m['source']}, p.{m['page']}")

if __name__ == "__main__":
    print("논문 Q&A 챗봇 (종료: Ctrl+C)\n")
    while True:
        q = input("질문: ").strip()
        if q:
            query(q)
        print()
