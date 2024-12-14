import streamlit as st

st.title("Private GPT")import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain.vectorstores import Chroma
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.callbacks.base import BaseCallbackHandler
from langchain.memory import ConversationSummaryBufferMemory

st.set_page_config(
    page_title="Priavte GPT",
    page_icon="🤣",
)


class ChatCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        self.response = ""

    def on_llm_start(self, *arg, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *arg, **kwargs):
        save_message(self.response, "ai")

    def on_llm_new_token(self, token, *arg, **kwargs):
        self.response += token
        self.message_box.markdown(self.response)


llm = ChatOpenAI(
    temperature=0.1,
    streaming=True,
    callbacks=[ChatCallbackHandler()],
)

memory_llm = ChatOpenAI(temperature=0.1)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a helpful assistant. Answer questions using only the following context.
            and You remember conversations with human.
            If you don't know the answer just say you don't know, dont't makt it
            ------
            {context}            
            """,
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)

@st.cache_resource(show_spinner="Embedding file...")
def embed_file(file):
    file_name = file.name
    file_path = f"./.cache/private_files/{file_name}"
    file_context = file.read()
    with open(file_path, "wb") as f:
        f.write(file_context)
    loader = UnstructuredFileLoader(file_path)
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n\n",
        chunk_size=500,
        chunk_overlap=60,
    )
    documents = loader.load_and_split(text_splitter=splitter)
    cache_dir = LocalFileStore(f"./.cache/private_embeddings/{file_name}")
    embedder = OpenAIEmbeddings()
    cache_embedder = CacheBackedEmbeddings.from_bytes_store(embedder, cache_dir)
    vectorStore = Chroma.from_documents(documents, cache_embedder)
    retriever = vectorStore.as_retriever()
    return retriever

def paint_history():
    for dic_message in st.session_state["messages"]:
        send_message(dic_message["message"], dic_message["role"], save=False)


def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
        if save:
            save_message(message, role)

def format_doc(documents):
    return "\n\n".join(doc.page_content for doc in documents)

def memory_load(input):
    return memory.load_memory_variables({})["history"]


if "memory" not in st.session_state:
    st.session_state["memory"] = ConversationSummaryBufferMemory(
        llm=memory_llm,
        max_token_limit=150,
        memory_key="history",
        return_messages=True,
    )

memory = st.session_state["memory"]


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
        type=["txt", "pdf", "docx"],
    )

if file:
    retriever = embed_file(file)
    send_message("How can I help you?", "ai", save=False)
    paint_history()
    answer = st.chat_input("Ask anything about your file....")
    if answer:
        send_message(answer, "human")
        chain = (
            {
                "context": retriever | RunnableLambda(format_doc),
                "history": RunnableLambda(memory_load),
                "question": RunnablePassthrough(),
            }
            | prompt
            | llm
        )
        with st.chat_message("ai"):
            response = chain.invoke(answer)
            memory.save_context({"input": answer}, {"output": response.content})
else:
    st.session_state["messages"] = []
    memory.clear()