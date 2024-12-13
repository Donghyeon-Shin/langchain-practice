import streamlit as st
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda

st.set_page_config(
    page_title="Document GPT",
    page_icon="🤣",
)

llm = ChatOpenAI(temperature=0.1)


@st.cache_resource(show_spinner="Embedding file...")
def embed_file(file):
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=50,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    embedder = OpenAIEmbeddings()
    cached_embedding = CacheBackedEmbeddings.from_bytes_store(embedder, cache_dir)
    vectorStore = Chroma.from_documents(docs, cached_embedding)
    retriver = vectorStore.as_retriever()
    return retriver


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
        if save:
            st.session_state["messages"].append({"message": message, "role": role})


def paint_history():
    for message in st.session_state["messages"]:
        send_message(message["message"], message["role"], save=False)


def docs_format(docs):
    return ".\n.\n".join(doc.page_content for doc in docs)


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a helpful assistant. Answer questions using only the following context. 
            If you don't know the answer just say you don't know, dont't makt it up:
            ----------
            {context},
            """,
        ),
        ("human", "{question}"),
    ]
)

st.title("Document GPT")

st.markdown(
    """
Welcome!
            
Use this chatbot to ask questions to an AI about your files!
Upload your files on the sidebar.
"""
)

with st.sidebar:
    file = st.file_uploader(
        "Upload a .txt .pdf or .docx file",
        type=["pdf", "txt", "docx"],
    )

if file:
    retriever = embed_file(file)
    send_message("Hi Can I help you?", "ai", save=False)
    paint_history()
    question = st.chat_input("Ask anything about your file....")
    if question:
        send_message(question, "human")
        chain = (
            {
                "context": retriever | RunnableLambda(docs_format),
                "question": RunnablePassthrough(),
            }
            | prompt
            | llm
        )
        response = chain.invoke(question).content
        send_message(response, "ai")
else:
    st.session_state["messages"] = []
