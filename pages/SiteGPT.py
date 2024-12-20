import streamlit as st
from pages.SiteGPT.utils import paint_message, send_message
from pages.SiteGPT.data_process import get_retriever_in_website
from pages.SiteGPT.chain import invoke_chain, initialize_memory

st.set_page_config(
    page_title="Site GPT",
    page_icon="🤣",
)
st.title("Site GPT")

st.markdown(
    """ 
    Ask questions about the content of a website.

    Start by writing the URL of the website on the sidebar.
    """
)

# https://deepmind.google/sitemap.xml

with st.sidebar:
    url = st.text_input(
        "Write down a URL",
        placeholder="https://example.com/sitemap.xml",
    )

if url:
    if ".xml" not in url:
        with st.sidebar:
            st.error("Please write down a Stiemap URL")
    else:
        retriever = get_retriever_in_website(url)

        send_message(st.session_state["messages"], "How can I help you?", "ai", save=False)
        paint_message(st.session_state["messages"])
        question = st.chat_input("Ask any questions in the document!")
        if question:
            send_message(st.session_state["messages"], question, "human")
            invoke_chain(st.session_state["messages"], retriever, question)
else:
    st.session_state["messages"] = []
    initialize_memory()
