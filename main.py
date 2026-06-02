import os
import streamlit as st

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# =========================
# CONFIG
# =========================

load_dotenv()

MODEL = "llama-3.3-70b-versatile"
DB_NAME = "vector_db"

# =========================
# LOAD PDF
# =========================

loader = PyPDFLoader("./FastAPI_RAG_Test_Notes.pdf")
documents = loader.load()

# =========================
# CHUNKING
# =========================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

texts = text_splitter.split_documents(documents)

# =========================
# EMBEDDINGS
# =========================

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# =========================
# VECTOR STORE
# =========================

vectorstore = Chroma.from_documents(
    documents=texts,
    embedding=embeddings,
    persist_directory=DB_NAME
)

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 4}
)

# =========================
# LLM
# =========================

llm = ChatGroq(
    temperature=0,
    model_name=MODEL
)

# =========================
# PROMPT
# =========================

SYSTEM_PROMPT_TEMPLATE = """
You are a helpful AI assistant.

Answer the user's question only using the provided context.

If the answer is present in the context, provide a clear and concise response.

If the answer is not available in the context, say:
"I could not find the answer in the provided document."

Context:
{context}
"""

# =========================
# RAG FUNCTION
# =========================

def answer_question(question):

    docs = retriever.invoke(question)

    context = "\n\n".join(
        doc.page_content
        for doc in docs
    )

    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        context=context
    )

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=question)
    ])

    return response.content

# =========================
# STREAMLIT UI
# =========================

st.set_page_config(
    page_title="PDF RAG Chatbot",
    page_icon="📚"
)

st.title("📚 FastAPI PDF RAG Chatbot")

question = st.text_input(
    "Ask a question about the PDF"
)

if st.button("Ask"):

    if question:

        answer = answer_question(question)

        st.success(answer)