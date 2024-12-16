import streamlit as st
import json
from langchain.chat_models import ChatOpenAI
from langchain.retrievers import WikipediaRetriever
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.schema.runnable import RunnableLambda
from langchain.schema import BaseOutputParser


class JsonOutputParser(BaseOutputParser):
    def parse(self, text):
        text = text.replace("```", "").replace("json", "")
        return json.loads(text)

output_parser = JsonOutputParser()

st.set_page_config(
    page_title="QuizGpt",
    page_icon="🤣",
)

st.title("QuizGPT")

llm = ChatOpenAI(
    temperature=0.1,
    model="gpt-3.5-turbo-0125",
    streaming=True,
    # callbacks=[StreamingStdOutCallbackHandler()],
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

formatting_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a powerful formatting algorithm.

            You format exam question intop JSON format.
            Answers with (o) are the correct ones.

            Example Input:

            Question: What is the color of the occean?
            Answers: Red|Yellow|Green|Blue(o)

            Question: What is the capital or Georgia?
            Answers: Baku|Tbilisi(o)|Manila|Beirut

            Question: When was Avator released?
            Answers: 2007|2001|2009(o)|1998

            Question: Who was Julius Caesar?
            Answers: A Roman Emperor(o)|Painter|Actor|Model

            Example Output:
            
            ```json
            {{ "questions": [
                    {{
                        "question": "What is the color of the occean?",
                        "answers": [
                            {{
                                "answer": "Red",
                                "correct": false
                            }},
                            {{
                                "answer": "Yellow"
                                "correct": false
                            }},
                            {{
                                "answer": "Green",
                                "correct": false
                            }},
                            {{
                                "answer": "Blue",
                                "correct": true
                            }},
                        ]
                    }},
                    {{
                        "question": "What is the capital or Georgia?",
                        "answers": [
                            {{
                                "answer": "Baku",
                                "correct": false
                            }},
                            {{
                                "answer": "Tbilisi"
                                "correct": true
                            }},
                            {{
                                "answer": "Manila",
                                "correct": false
                            }},
                            {{
                                "answer": "Beirut",
                                "correct": false
                            }},
                        ]
                    }},
                    {{
                        "question": "When was Avator released?",
                        "answers": [
                            {{
                                "answer": "2007",
                                "correct": false
                            }},
                            {{
                                "answer": "2001"
                                "correct": false
                            }},
                            {{
                                "answer": "2009",
                                "correct": true
                            }},
                            {{
                                "answer": "1998",
                                "correct": false
                            }},
                        ]
                    }},
                    {{
                        "question": "Who was Julius Caesar?",
                        "answers": [
                            {{
                                "answer": "A Roman Emperor",
                                "correct": true
                            }},
                            {{
                                "answer": "Painter"
                                "correct": false
                            }},
                            {{
                                "answer": "Actor",
                                "correct": false
                            }},
                            {{
                                "answer": "Model",
                                "correct": false
                            }},
                        ]
                    }}                                                
                ]
            }}```

            Your turn!

            Question : {context}
            """,
        )
    ]
)

formatting_chain = formatting_prompt | llm


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

    start = st.button("Generate Quiz")
    if start:
        chain = {"context": questions_chain} | formatting_chain | output_parser
        response = chain.invoke(docs)
        st.write(response)
