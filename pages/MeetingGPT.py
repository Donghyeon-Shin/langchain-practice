import streamlit as st
import subprocess
import math
from pydub import AudioSegment
import openai
import glob
import os
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.output_parser import StrOutputParser
from langchain.storage import LocalFileStore
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.callbacks.base import BaseCallbackHandler

class ChatCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        self.response = ""

    def on_llm_start(self, *arg, **kwargs):
        self.message_box = st.empty()
    
    def on_llm_new_token(self, token, *arg, **kwargs):
        self.response += token
        self.message_box.markdown(self.response)

has_transcrible = os.path.exists("./.cache/meeting_files/chunks/Bible_summary.txt")

llm = ChatOpenAI(
    temperature=0.1,
)

choose_llm = ChatOpenAI(
    temperature=0.1,
    streaming=True,
    callbacks=[ChatCallbackHandler()]
)

splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=1000,
    chunk_overlap=100,
)


@st.cache_data()
def embed_file(file_path, file_name):
    loader = TextLoader(file_path)
    documents = loader.load_and_split(text_splitter=splitter)
    cache_dir = LocalFileStore(f"./.cache/meeting_files/embeddings/{file_name}")
    embedder = OpenAIEmbeddings()
    cache_embedder = CacheBackedEmbeddings.from_bytes_store(embedder, cache_dir)
    vectorStore = FAISS.from_documents(documents, cache_embedder)
    retriever = vectorStore.as_retriever()
    return retriever


@st.cache_data()
def transcribe_chunks(chunk_folder, destination):
    if has_transcrible:
        return
    files = glob.glob(f"{chunk_folder}/*.mp3")
    files.sort()
    for file in files:
        with open(file, "rb") as audio_file, open(destination, "a") as text_file:
            transcipts = openai.Audio.transcribe(
                "whisper-1",
                audio_file,
                language="ko",
            )
            text_file.write(transcipts["text"])


@st.cache_data()
def extract_audio_from_video(video_path, audio_path):
    if has_transcrible:
        return
    command = [
        "ffmpeg",
        "-i",
        video_path,
        "-vn",
        audio_path,
    ]
    subprocess.run(command)


@st.cache_data()
def cut_audio_in_chunks(audio_path, chunk_size, chunks_folder):
    if has_transcrible:
        return
    track = AudioSegment.from_mp3(audio_path)
    chunk_overlap = 10 * 1000  # overlap_size = 10 seconds
    chunk_len = chunk_size * 60 * 1000 - chunk_overlap
    chunks = math.ceil(len(track) / chunk_len)
    for i in range(chunks):
        start_time = i * chunk_len
        end_time = (i + 1) * chunk_len + chunk_overlap
        chunk = track[start_time:end_time]
        chunk.export(f"./{chunks_folder}/chunk_{i}.mp3", format="mp3")


def get_answers(inputs):
    docs = inputs["docs"]
    question = inputs["question"]
    answer_prompt = ChatPromptTemplate.from_template(
        """
            Using ONLY the following context answer the user's question. If you can't answer,
            JUST say you don't know. don't make anything up.

            Then, give a score to the answer between 0 and 5. 0 being not helpful to the user and 5 being helpful to the user.

            Make sure to include the answer's score.
            ONLY one result should be output.

            Content : {context}

            Examples:

            Question: How far away the moon?
            Answer: The moon is 384,400 km away.
            Score: 5

            Question: How far away is the sun?
            Answer: I don't know
            Score: 0

            Your turn!

            Question : {question}
            """
    )

    answer_chain = answer_prompt | llm | StrOutputParser()

    return {
        "question": question,
        "answers": [
            answer_chain.invoke(
                {
                    "context": doc.page_content,
                    "question": question,
                }
            )
            for doc in docs
        ],
    }


def choose_answer(inputs):
    question = inputs["question"]
    answers = inputs["answers"]

    format_answers = "\n\n".join(answer for answer in answers)

    choose_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                Use ONLY the following pre-existing answers to the user's question.

                Use the answers that have the highest score (more helpful) and favor the most recent ones.

                Return the sources of the answers as they are, do not change them.

                You must print out only one answer. and Don't print out the score

                Answer: {answers}
                """,
            ),
            ("human", "{question}"),
        ]
    )

    choose_chain = choose_prompt | choose_llm | StrOutputParser()

    respose = choose_chain.invoke({"answers" : format_answers, "question" : question})
    return respose


st.set_page_config(
    page_title="MettingGPT",
    page_icon="🤣",
)

st.title("MeetingGPT")

st.markdown(
    """
    Welcome to MettingGPT, upload a video and I will give you a transcript, a summary and a chat bot to ask any question about it.

    Get started by uploading a video file in the sidebar.
    """
)

with st.sidebar:
    video = st.file_uploader("Video", type=["mp4", "avi", "mkv", "mov"])
if video:
    with st.status("Loading video....") as status:
        video_content = video.read()
        video_path = f"./.cache/meeting_files/{video.name}"
        audio_path = video_path.replace("mp4", "mp3")
        chuck_path = "./.cache/meeting_files/chunks"
        transcribe_path = video_path.replace("mp4", "txt")
        with open(video_path, "wb") as f:
            f.write(video_content)
        status.update(label="Extracting audio....")
        extract_audio_from_video(video_path, audio_path)
        status.update(label="Cutting audio segments....")
        cut_audio_in_chunks(audio_path, 10, chuck_path)
        status.update(label="Transcribing audio....")
        transcribe_chunks(chuck_path, transcribe_path)

    transcript_tabs, summary_tabs, qa_tab = st.tabs(
        [
            "Transcript",
            "Summary",
            "Q&A",
        ]
    )

    with transcript_tabs:
        with open(transcribe_path, "r") as file:
            st.write(file.read())

    with summary_tabs:
        summary_start_button = st.button("Generate Summary")

        if summary_start_button:
            loader_path = "./.cache/meeting_files/Bible_small_summary.txt"
            loader = TextLoader(loader_path)
            docs = loader.load_and_split(text_splitter=splitter)

            first_summary_prompt = ChatPromptTemplate.from_template(
                """
                Write a concise summary of the following:
                "{text}"
                CONCISE SUMMARY:
                """
            )

            first_summary_chain = first_summary_prompt | llm | StrOutputParser()
            summary = first_summary_chain.invoke({"text": docs[0].page_content})

            refine_prompt = ChatPromptTemplate.from_template(
                """
                Your job is to produce a final summary.
                We have provided an existing summary up to a certain point: {existing_summary}
                We have the opportunity to refine the existing summary (only if needed) with some more context below.
                ---------
                {context}
                ---------
                Given the new context, refine the original summary.
                If the context isn't useful, RETURN the original summary.
                """
            )

            refine_chain = refine_prompt | llm | StrOutputParser()

            with st.status("Summarizing") as status:
                for i, doc in enumerate(docs[1:]):
                    status.update(label=f"Processing document {i+1}/{len(docs)-1}")
                    summary = refine_chain.invoke(
                        {
                            "existing_summary": summary,
                            "context": doc.page_content,
                        }
                    )
                st.write(summary)
            st.write(summary)

    with qa_tab:
        retriever = embed_file(transcribe_path, video.name)

        docs = retriever.invoke("Dose he talk about the bible?")

        question = st.text_input("Answer anyting about the video")

        if question:
            with st.chat_message("ai"):
                research_chain = (
                    {
                        "docs": retriever,
                        "question": RunnablePassthrough(),
                    }
                    | RunnableLambda(get_answers)
                    | RunnableLambda(choose_answer)
                )

                research_chain.invoke(question)
