import streamlit as st
from datetime import datetime

today = datetime.today().strftime("%H:%M:%S")

st.title(today)

model = st.selectbox(
    "Choose your model",
    ("GPT-3", "GPT-4"),
)

if model == "GPT-3":
    st.write("cheep")
else:
    st.write("not cheep")

name = st.text_input("What is your name?")
st.write(name)

value = st.slider(
    "htemperature",
    min_value=0.1,
    max_value=1.0,
)

st.write(value)