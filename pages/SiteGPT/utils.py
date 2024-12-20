import streamlit as st

def paint_message(messages):
    for message in messages:
        send_message(messages, message["message"], message["role"], save=False)


def save_message(messages, message, role):
    messages.append({"message": message, "role": role})


def send_message(messages, message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
        if save:
            save_message(messages, message, role)
