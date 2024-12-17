import streamlit as st
import json
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

format_function = {
    "name": "formatting_quiz",
    "description": "function that takes a list of questions and answers and returns a quiz",
    "parameters": {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                        },
                        "answers": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "answer": {
                                        "type": "string",
                                    },
                                    "correct": {
                                        "type": "boolean",
                                    },
                                },
                                "required": ["answer", "correct"],
                            },
                        },
                    },
                    "required": ["question", "answers"],
                },
            },
        },
        "required": ["questions"],
    },
}

llm = ChatOpenAI(
    temperature=0.1,
    model="gpt-3.5-turbo-0125",
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()],
).bind(
    function_call={
        "name": "formatting_quiz",
    },
    functions=[
        format_function,
    ],
)


def format_docs(documents):
    return "\n\n".join(doc.page_content for doc in documents)


questions_prompt = ChatPromptTemplate.from_messages(
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
            Answers: Red|Yellow|Green|Blue(o)

            Question: What is the capital or Georgia?
            Answers: Baku|Tbilisi(o)|Manila|Beirut

            Question: When was Avator released?
            Answers: 2007|2001|2009(o)|1998 
            
            Question: Who was Julius Caesar?
            Answers: A Roman Emperor(o)|Painter|Actor|Model
            
            Your turn!
            Context: {context}
            """,
        )
    ]
)

questions_chain = {"context": RunnableLambda(format_docs)} | questions_prompt | llm


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


@st.cache_data(show_spinner="Making quiz...")
def run_quiz_chain(_docs, topic):
    response = questions_chain.invoke(_docs)
    questions_json = json.loads(
        response.additional_kwargs["function_call"]["arguments"]
    )
    return questions_json


@st.cache_data(show_spinner="Searching wikipedia...")
def wiki_search(topic):
    retriever = WikipediaRetriever(top_k_results=5)
    docs = retriever.get_relevant_documents(topic)
    return docs


with st.sidebar:
    docs = None
    topic = None
    file = None
    source_options = st.selectbox(
        "Choose what you want to use.",
        (
            "File",
            "Wikipedia Article",
        ),
    )
    if source_options == "File":
        file = st.file_uploader(
            "Upload a .docx, .txt, .pdf file", type=["pdf", "txt", "docx"]
        )
        if file:
            docs = split_file(file)
    else:
        topic = st.text_input("Search Wikipedia")
        if topic:
            docs = wiki_search(topic)

if not docs:
    st.markdown(
        """
        Welcome to QuizGPT.

        I will make a quiz from Wikipedia articles or files you upload to test your knowledge and help you study.

        Get Started by uploading a file or searching on Wikipedia in the sidebar.
        """
    )
    st.session_state["start_status"] = False
else:
    questions_json = run_quiz_chain(docs, topic if topic else file.name)

    start_button = st.empty()

    if not st.session_state["start_status"]:
        start_button = st.button("Quiz Start!")
        with st.sidebar:
            chance_cnt = st.selectbox("Choose the number of chances", options=[1, 2, 3])
            st.session_state["chance_cnt"] = chance_cnt
            st.session_state["button_name"] = "submit"

    if start_button:
        st.session_state["start_status"] = True

    if st.session_state["start_status"]:
        with st.form("questions_form"):
            chance_cnt = st.session_state["chance_cnt"]
            if chance_cnt > 1:
                st.write(f"You have {chance_cnt} chances left.")
            elif chance_cnt == 1:
                st.write(f"You have {chance_cnt} chance left.")
            else:
                st.write("You don't have a chance left. Check the answer")

            for question in questions_json["questions"]:
                st.write(question["question"])
                value = st.radio(
                    "Select an answer",
                    [answer["answer"] for answer in question["answers"]],
                    index=None,
                )
                if {"answer": value, "correct": True} in question["answers"]:
                    st.success("Correct!")
                elif value is not None:
                    if st.session_state["chance_cnt"] == 0:
                        for answer in question["answers"]:
                            if answer["correct"] == True:
                                correct_answer = answer["answer"]
                                st.error(f"answer is {correct_answer}")
                    else:
                        st.error("Wrong!")
            submit_button = st.form_submit_button(st.session_state["button_name"])

            if st.session_state["chance_cnt"] > 0:
                st.session_state["chance_cnt"] = st.session_state["chance_cnt"] - 1
                if st.session_state["chance_cnt"] == 0:
                    st.session_state["button_name"] = "finish quiz"
            else:
                st.session_state["start_status"] = False
