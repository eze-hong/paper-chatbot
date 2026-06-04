import streamlit as st
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

from config import LLM_MODEL, TOP_K, COLLECTION_NAME
from embeddings import get_embedding_function
from db import get_collection

load_dotenv()

st.set_page_config(
    page_title="논문 Q&A 챗봇",
    page_icon="📄",
    layout="centered",
)

PROMPT_TEMPLATE = PromptTemplate.from_template(
    "당신은 논문 내용을 기반으로 질문에 답변하는 AI입니다.\n"
    "아래 컨텍스트만 사용하여 답변하고, 모르면 모른다고 하세요.\n\n"
    "[컨텍스트]\n{context}\n\n"
    "[질문]\n{question}"
)


@st.cache_resource
def load_embed_fn():
    return get_embedding_function()

@st.cache_resource
def load_collection():
    return get_collection()

@st.cache_resource
def load_llm():
    return ChatGoogleGenerativeAI(model=LLM_MODEL)


def run_query(question: str, embed_fn, collection, llm):
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
    raw = llm.invoke(prompt)
    answer = raw.content[0]["text"] if isinstance(raw.content, list) else raw.content
    sources = list({f"{m['source']} p.{m['page']}" for m in metas})
    return answer, sources


# 사이드바
with st.sidebar:
    st.title("⚙️ 설정")
    st.info(
        f"**컬렉션**: {COLLECTION_NAME}\n\n"
        f"**검색 청크 수**: {TOP_K}\n\n"
        f"**LLM**: {LLM_MODEL}"
    )
    if st.button("대화 초기화"):
        st.session_state.messages = []
        st.rerun()

# 메인 영역
st.title("📄 논문 Q&A 챗봇")
st.caption("업로드된 논문 PDF를 기반으로 질문에 답변합니다.")

try:
    with st.status("리소스 로딩 중...", expanded=True) as status:
        st.write("① 임베딩 모델 로딩 중... (ko-sroberta)")
        embed_fn = load_embed_fn()
        st.write("② 벡터 DB 연결 중... (ChromaDB)")
        collection = load_collection()
        st.write("③ LLM 초기화 중... (Gemini)")
        llm = load_llm()
        status.update(label="로딩 완료!", state="complete", expanded=False)
except Exception as e:
    st.error(f"리소스 로딩 실패: {e}")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

# 대화 히스토리 렌더링
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("📎 출처 보기"):
                for src in msg["sources"]:
                    st.markdown(f"- {src}")

# 새 질문 처리
if question := st.chat_input("논문에 대해 질문하세요..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("논문에서 검색 중..."):
            answer, sources = run_query(question, embed_fn, collection, llm)
        st.markdown(answer)
        if sources:
            with st.expander("📎 출처 보기"):
                for src in sources:
                    st.markdown(f"- {src}")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
    })
