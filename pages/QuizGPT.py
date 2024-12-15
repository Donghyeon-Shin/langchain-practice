import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.retrievers import WikipediaRetriever
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.schema.runnable import RunnableLambda

st.set_page_config(
    page_title="QuizGpt",
    page_icon="🤣",
)

st.title("QuizGPT")

llm = ChatOpenAI(
    temperature=0.1,
    model="gpt-3.5-turbo-0125",
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()],
)


@st.cache_resource(show_spinner="Loading file...")
def split_file(file):
    file_name = file.name
    file_path = f"./.cache/quiz_files/{file_name}"
    file_context = file.read()
    with open(file_path, "wb") as f:
        f.write(file_context)
    loader = UnstructuredFileLoader(file_path)
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    documents = loader.load_and_split(text_splitter=splitter)
    return documents

def format_docs(documents):
    return "\n\n".join(doc.page_content for doc in documents)


with st.sidebar:
    docs = None
    choice = st.selectbox(
        "Choose what you want to use.",
        (
            "File",
            "Wikipedia Article",
        ),
    )
    if choice == "File":
        file = st.file_uploader(
            "Upload a .docx, .txt, .pdf file", type=["pdf", "txt", "docx"]
        )
        if file:
            docs = split_file(file)
    else:
        topic = st.text_input("Search Wikipedia")
        if topic:
            retriever = WikipediaRetriever(top_k_results=5)
            with st.status("Searching wikipedia....."):
                docs = retriever.get_relevant_documents(topic)

if not docs:
    st.markdown(
        """
        Welcome to QuizGPT.

        I will make a quiz from Wikipedia articles or files you upload to test your knowledge and help you study.

        Get Started by uploading a file or searching on Wikipedia in the sidebar.
        """
    )
else:
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                You are a helpful assistant that is role playing as a teacher.

                Based ONLY on the following context make 10 questoins to test the user's knowledge about the text.

                Each question should have 4 answers, three of them must be incorrect and one should be correct.

                Use (o) to signal the correct answer.

                Questoin examples

                Question: What is the color of the occean?
                Answers: Red/Yellow/Green/Blue(o)

                Question: What is the capital or Georgia?
                Answers: Baku/Tbilisi(o)/Manila/Beirut

                Question: When was Avator released?
                Answers: 2007/2001/2009(o)/1998

                Question: Who was Julius Caesar?
                Answers: A Roman Emperor(o)/Painter/Actor/Model

                Your turn!

                Context: {context}
                """,
            )
        ]
    )

    start = st.button("Generate Quiz")
    if start:
        chain.invoke(docs)
