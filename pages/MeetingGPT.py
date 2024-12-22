import streamlit as st
import subprocess
import math
from pydub import AudioSegment
import openai
import glob
import os

has_transcrible = os.path.exists("./.cache/meeting_files/chunks/Bible_summary.txt")

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
    with st.status("Loading video...."):
        video_content = video.read()
        video_path = f"./.cache/meeting_files/{video.name}"
        audio_path = video_path.replace("mp4", "mp3")
        chuck_path = "./.cache/meeting_files/chunks"
        transcribe_path = video_path.replace("mp4", "txt")
        with open(video_path, "wb") as f:
            f.write(video_content)
    with st.status("Extracting audio...."):
        extract_audio_from_video(video_path, audio_path)
    with st.status("Cutting audio segments...."):
        cut_audio_in_chunks(audio_path, 10, chuck_path)
    with st.status("Transcribing audio...."):
        transcribe_chunks(chuck_path, transcribe_path)
