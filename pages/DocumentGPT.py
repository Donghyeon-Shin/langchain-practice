import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain.vectorstores import Chroma
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.callbacks.base import BaseCallbackHandler
from langchain.globals import set_llm_cache
from langchain.memory import ConversationSummaryBufferMemory

st.set_page_config(
    page_title="Document GPT",
    page_icon="🤣",
)


class ChatCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        self.message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)


llm = ChatOpenAI(
    temperature=0.1,
    streaming=True,
    callbacks=[
        ChatCallbackHandler(),
    ],
)

memory_llm = ChatOpenAI(temperature=0.1)

chat_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a helpful assistant. Answer questions using only the following context.
            and You remember conversations with human.
            If you don't know the answer just say you don't know, dont't makt it up:
            -----
            {context}
            """,
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)

if "memory" not in st.session_state:
    st.session_state["memory"] = ConversationSummaryBufferMemory(
        llm=memory_llm,
        max_token_limit=150,
        memory_key="chat_history",
        return_messages=True,
    )

memory = st.session_state["memory"]

@st.cache_resource(show_spinner="Embedding file...")
def embed_file(file):
    file_name = file.name
    file_content = file.read()
    file_path = f"./.cache/files/{file_name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    cache_dir = LocalFileStore(f"./cache/embeddings/{file_name}")
    loader = UnstructuredFileLoader(file_path)
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n\n",
        chunk_size=500,
        chunk_overlap=60,
    )
    documents = loader.load_and_split(text_splitter=splitter)
    embeder = OpenAIEmbeddings()
    cache_embedder = CacheBackedEmbeddings.from_bytes_store(embeder, cache_dir)
    vectorStore = Chroma.from_documents(documents, cache_embedder)
    retriever = vectorStore.as_retriever()
    return retriever


def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
        if save:
            save_message(message, role)


def paint_history():
    for message_dic in st.session_state["messages"]:
        send_message(message_dic["message"], message_dic["role"], save=False)


def format_doc(documents):
    return "\n\n".join(doc.page_content for doc in documents)


def memory_load(inputs):
    return memory.load_memory_variables({})["chat_history"]


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
    send_message("How can I help you?", "ai", save=False)
    paint_history()
    question = st.chat_input("Ask anything about your file....")
    if question:
        send_message(question, "human")
        chain = (
            {
                "context": retriever | RunnableLambda(format_doc),
                "chat_history": RunnableLambda(memory_load),
                "question": RunnablePassthrough(),
            }
            | chat_prompt
            | llm
        )
        with st.chat_message("ai"):
            response = chain.invoke(question)
        memory.save_context({"input": question}, {"output": response.content})
else:
    st.session_state["messages"] = []
    memory.clear()
